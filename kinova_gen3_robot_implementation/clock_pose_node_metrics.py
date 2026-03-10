#!/usr/bin/env python3
"""
clock_pose_node_metrics.py (ROS 2 Jazzy)

Standalone perception node for the clock pick task.
Detects a digital alarm clock via Roboflow, estimates its 6-DoF pose
(position + yaw via 3D PCA on depth-projected bbox points), and publishes:

    /clock/pose_base      PoseStamped   (position + yaw-only orientation in base_link)
    /clock/quality        Float32MultiArray [conf, pose_mode, tf_ok, valid_ratio, bbox_area, depth_med]
    /clock/markers        Marker        (CUBE at clock centroid with yaw orientation)
    /clock_detection/debug_image  Image (annotated BGR)

The supervisor (clock_pick_supervisor_metrics.py) subscribes to /clock/pose_base
and /clock/quality to decide when the clock is stably detected and where to grasp.

Run:
  ros2 run bottle_grasping clock_pose_node_metrics --ros-args \\
    -p trial_id:=1 -p variant:=commit_only -p log_dir:=/tmp/clock_logs
"""

import os
import csv
import math
import time
import base64
import threading

import numpy as np
import cv2
import requests

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge

import message_filters
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose_stamped


# ============================
# Roboflow config
# ============================
ROBOFLOW_API_KEY_DEFAULT = "b9dqf9zSgloKnDbz67tH"
ROBOFLOW_PROJECT_DEFAULT = "clock-geqo0"
ROBOFLOW_VERSION_DEFAULT = "1"
ROBOFLOW_CONFIDENCE_DEFAULT = 0.40

ROBOFLOW_TIMEOUT_S = 2.5
ROBOFLOW_RETRIES = 1

# ============================
# Clock dimensions
# ============================
INCH_TO_M = 0.0254
CLOCK_LENGTH_M = 6.0 * INCH_TO_M    # 0.1524 m — long axis
CLOCK_WIDTH_M  = 1.5 * INCH_TO_M    # 0.0381 m — depth (front-to-back)
CLOCK_HEIGHT_M = 3.0 * INCH_TO_M    # 0.0762 m — vertical height
GRASP_END_INSET_M = 0.5 * INCH_TO_M  # 0.0127 m — inset from clock long-axis end

# ============================
# Defaults
# ============================
BASE_FRAME = "base_link"
CAM_FRAME  = "camera_color_frame"

RGB_TOPIC   = "/camera/color/image_raw"
DEPTH_TOPIC = "/camera/depth/image_raw"
INFO_TOPIC  = "/camera/color/camera_info"

PROCESS_HZ = 5.0

TF_LOOKUP_TIMEOUT_S = 0.25
TF_WARN_THROTTLE_S  = 1.0

DEPTH_MIN_M = 0.15
DEPTH_MAX_M = 2.50

# EMA smoothing
POS_ALPHA  = 0.35
YAW_ALPHA  = 0.35

# Workspace bounds
WS_MIN_X, WS_MAX_X = -0.75, 0.90
WS_MIN_Y, WS_MAX_Y = -0.60, 1.00
WS_MIN_Z, WS_MAX_Z = 0.00, 0.90

# Staleness
STALE_FRAMES = 15   # ~3s at 5 Hz
MAX_JUMP_M   = 0.12
JUMP_RESET_M = 0.25

# PCA yaw extraction
PCA_SUBSAMPLE_STEP = 5
PCA_MIN_POINTS = 20

# Marker colors
CLOCK_COLOR = (0.1, 0.7, 1.0, 0.7)

# Logging
LOG_EVERY_N = 10


# ============================
# Helpers
# ============================

def depth_to_meters(depth: np.ndarray, encoding: str = "") -> np.ndarray:
    d = depth.astype(np.float32)
    enc_lower = encoding.lower()
    if enc_lower in ("16uc1", "mono16"):
        d /= 1000.0
    elif enc_lower in ("32fc1",):
        pass
    elif depth.dtype in (np.uint16,):
        d /= 1000.0
    return d


def clip_bbox(x1, y1, x2, y2, W, H):
    x1c = int(max(0, min(W - 1, x1)))
    x2c = int(max(0, min(W - 1, x2)))
    y1c = int(max(0, min(H - 1, y1)))
    y2c = int(max(0, min(H - 1, y2)))
    if x2c <= x1c or y2c <= y1c:
        return None
    return x1c, y1c, x2c, y2c


def bbox_from_pred(p):
    cx, cy = float(p["x"]), float(p["y"])
    w, h = float(p["width"]), float(p["height"])
    x1 = int(cx - w / 2.0)
    y1 = int(cy - h / 2.0)
    x2 = int(cx + w / 2.0)
    y2 = int(cy + h / 2.0)
    return x1, y1, x2, y2, int(cx), int(cy)


# ============================
# Roboflow client (threaded)
# ============================

class RoboflowClient:
    """Threaded Roboflow client so ROS callbacks never block on HTTP."""

    def __init__(self, logger, api_key: str, project: str, version: str, conf: float):
        self.logger = logger
        self.api_key = api_key
        self.project = project
        self.version = version
        self.conf = float(conf)
        self.session = requests.Session()
        self.api_url = (
            f"https://detect.roboflow.com/{self.project}/{self.version}"
            f"?api_key={self.api_key}"
            f"&confidence={int(self.conf * 100)}"
        )
        self.logger.info(
            f"[Roboflow] Using project={self.project} version={self.version} "
            f"conf={self.conf:.2f}"
        )
        self._lock = threading.Lock()
        self._pending_bgr = None
        self._latest_result: list = []
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._stop = threading.Event()
        self._has_work = threading.Event()
        self._thread.start()

    def submit_and_get_latest(self, bgr: np.ndarray) -> list:
        with self._lock:
            self._pending_bgr = bgr
            result = list(self._latest_result)
        self._has_work.set()
        return result

    def _worker_loop(self):
        while not self._stop.is_set():
            self._has_work.wait(timeout=1.0)
            self._has_work.clear()
            with self._lock:
                bgr = self._pending_bgr
                self._pending_bgr = None
            if bgr is None:
                continue
            preds = self._detect_sync(bgr)
            with self._lock:
                self._latest_result = preds

    def _detect_sync(self, bgr: np.ndarray) -> list:
        _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        last_err = None
        for _ in range(ROBOFLOW_RETRIES + 1):
            try:
                resp = self.session.post(
                    self.api_url,
                    data=img_b64,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    timeout=ROBOFLOW_TIMEOUT_S,
                )
                if resp.status_code != 200:
                    self.logger.error(
                        f"[Roboflow] HTTP {resp.status_code}: {resp.text[:200]}")
                    return []
                data = resp.json()
                preds = data.get("predictions", [])
                out = []
                for p in preds:
                    out.append({
                        "cls": str(p.get("class", "")),
                        "conf": float(p.get("confidence", 0.0)),
                        "x": float(p.get("x", 0.0)),
                        "y": float(p.get("y", 0.0)),
                        "width": float(p.get("width", 0.0)),
                        "height": float(p.get("height", 0.0)),
                    })
                return out
            except Exception as e:
                last_err = e
        self.logger.warn(f"[Roboflow] request failed after retries: {last_err}")
        return []

    def shutdown(self):
        self._stop.set()
        self._has_work.set()
        self._thread.join(timeout=3.0)


# ============================
# Main node
# ============================

class ClockPoseNodeMetrics(Node):
    def __init__(self, rf_api_key, rf_project, rf_version, rf_conf):
        super().__init__("clock_pose_node_metrics")
        self.rf_conf_threshold = float(rf_conf)

        # ── Optional CSV logging ──
        self.declare_parameter("trial_id", "")
        self.declare_parameter("variant", "")
        self.declare_parameter("log_dir", "")
        self.declare_parameter("enable_quality_csv", False)

        self._log_trial_id = str(self.get_parameter("trial_id").value)
        self._log_variant = str(self.get_parameter("variant").value)
        self._log_dir = str(self.get_parameter("log_dir").value)
        self._enable_quality_csv = bool(self.get_parameter("enable_quality_csv").value)

        self._t0_mono = time.monotonic()
        self._quality_csv_path = ""
        if self._enable_quality_csv and self._log_dir.strip():
            os.makedirs(self._log_dir, exist_ok=True)
            self._quality_csv_path = os.path.join(self._log_dir, "quality_stream.csv")
            if not os.path.exists(self._quality_csv_path):
                with open(self._quality_csv_path, "w", newline="") as f:
                    csv.writer(f).writerow([
                        "t_s", "trial_id", "variant",
                        "clock_conf", "pose_mode", "tf_ok",
                        "valid_ratio", "bbox_area", "depth_med",
                    ])

        # ── Configurable parameters ──
        self.declare_parameter("base_frame", BASE_FRAME)
        self.declare_parameter("camera_frame", CAM_FRAME)
        self.declare_parameter("rgb_topic", RGB_TOPIC)
        self.declare_parameter("depth_topic", DEPTH_TOPIC)
        self.declare_parameter("camera_info_topic", INFO_TOPIC)
        self.declare_parameter("roboflow_class_name", "clock")
        self.declare_parameter("min_depth_m", DEPTH_MIN_M)
        self.declare_parameter("max_depth_m", DEPTH_MAX_M)
        self.declare_parameter("workspace_min_x_m", WS_MIN_X)
        self.declare_parameter("workspace_max_x_m", WS_MAX_X)
        self.declare_parameter("workspace_min_y_m", WS_MIN_Y)
        self.declare_parameter("workspace_max_y_m", WS_MAX_Y)
        self.declare_parameter("workspace_min_z_m", WS_MIN_Z)
        self.declare_parameter("workspace_max_z_m", WS_MAX_Z)
        self.declare_parameter("grasp_end_inset_m", GRASP_END_INSET_M)
        self.declare_parameter("yaw_offset_rad", 0.0)
        self.declare_parameter("min_grasp_z_m", 0.03)

        self.base_frame = str(self.get_parameter("base_frame").value)
        self.camera_frame = str(self.get_parameter("camera_frame").value)
        self.rgb_topic = str(self.get_parameter("rgb_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self.roboflow_class_name = str(
            self.get_parameter("roboflow_class_name").value).strip().lower()
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)
        self.ws_min_x = float(self.get_parameter("workspace_min_x_m").value)
        self.ws_max_x = float(self.get_parameter("workspace_max_x_m").value)
        self.ws_min_y = float(self.get_parameter("workspace_min_y_m").value)
        self.ws_max_y = float(self.get_parameter("workspace_max_y_m").value)
        self.ws_min_z = float(self.get_parameter("workspace_min_z_m").value)
        self.ws_max_z = float(self.get_parameter("workspace_max_z_m").value)
        self.grasp_end_inset_m = float(self.get_parameter("grasp_end_inset_m").value)
        self.yaw_offset_rad = float(self.get_parameter("yaw_offset_rad").value)
        self.min_grasp_z_m = float(self.get_parameter("min_grasp_z_m").value)

        self.bridge = CvBridge()
        self.rf = RoboflowClient(
            self.get_logger(), rf_api_key, rf_project, rf_version, rf_conf)

        # ── Intrinsics ──
        self.fx = self.fy = self.cx = self.cy = None
        self.create_subscription(CameraInfo, self.camera_info_topic, self._on_info, 10)

        # ── TF ──
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._last_tf_warn = 0.0

        # ── Publishers ──
        self.pose_pub = self.create_publisher(PoseStamped, "/clock/pose_base", 10)
        self.quality_pub = self.create_publisher(Float32MultiArray, "/clock/quality", 10)
        self.marker_pub = self.create_publisher(Marker, "/clock_pick/clock_marker", 10)
        self.grasp_corner_pub = self.create_publisher(Marker, "/clock_pick/grasp_target_marker", 10)
        self.debug_image_pub = self.create_publisher(Image, "/clock_detection/debug_image", 10)

        # ── Synced RGB + Depth ──
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        rgb_sub = message_filters.Subscriber(self, Image, self.rgb_topic, qos_profile=qos)
        depth_sub = message_filters.Subscriber(self, Image, self.depth_topic, qos_profile=qos)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=5, slop=0.10)
        self.sync.registerCallback(self._on_rgb_depth)

        # ── Rate limiter ──
        self.last_t = 0.0
        self.period = 1.0 / max(1e-6, PROCESS_HZ)

        # ── EMA filter state ──
        self.have_clock = False
        self.clock_xyz = np.zeros(3, dtype=np.float64)
        self.clock_yaw = 0.0
        self.clock_miss_count = 0

        self._log_count = 0
        self._last_reject_log_t = 0.0
        self.get_logger().info("ClockPoseNodeMetrics running.")

    def _log_reject(self, reason: str):
        now = time.time()
        if (now - self._last_reject_log_t) < 1.0:
            return
        self._last_reject_log_t = now
        self.get_logger().warn(f"[CLOCK] rejected detection: {reason}")

    # ── Camera info ──
    def _on_info(self, msg: CameraInfo):
        if self.fx is None:
            self.fx = float(msg.k[0])
            self.fy = float(msg.k[4])
            self.cx = float(msg.k[2])
            self.cy = float(msg.k[5])
            self.get_logger().info(
                f"Intrinsics: fx={self.fx:.2f} fy={self.fy:.2f} "
                f"cx={self.cx:.2f} cy={self.cy:.2f}")

    # ── TF lookup ──
    def _lookup_base_T_cam(self):
        now_s = time.time()
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, self.camera_frame, rclpy.time.Time(),
                timeout=Duration(seconds=TF_LOOKUP_TIMEOUT_S))
            return tf
        except Exception as e:
            if (now_s - self._last_tf_warn) > TF_WARN_THROTTLE_S:
                self.get_logger().warn(
                    f"TF lookup failed ({self.base_frame} <- {self.camera_frame}): {e}")
                self._last_tf_warn = now_s
            return None

    def _transform_point(self, x_c, y_c, z_c, header, tf_base_cam):
        """Transform a camera-frame point to base_link using an already-looked-up TF."""
        ps = PoseStamped()
        ps.header = header
        ps.header.frame_id = self.camera_frame
        ps.pose.position.x = float(x_c)
        ps.pose.position.y = float(y_c)
        ps.pose.position.z = float(z_c)
        ps.pose.orientation.w = 1.0
        try:
            pw = do_transform_pose_stamped(ps, tf_base_cam)
            return float(pw.pose.position.x), float(pw.pose.position.y), float(pw.pose.position.z)
        except Exception:
            return None

    # ── Marker ──
    def _publish_clock_marker(self, stamp_msg, xyz, yaw):
        mk = Marker()
        mk.header.frame_id = self.base_frame
        mk.header.stamp = stamp_msg
        mk.ns = "clock"
        mk.id = 1
        mk.type = Marker.CUBE
        mk.action = Marker.ADD
        mk.pose.position.x = float(xyz[0])
        mk.pose.position.y = float(xyz[1])
        mk.pose.position.z = float(xyz[2])
        half = float(yaw) / 2.0
        mk.pose.orientation.z = math.sin(half)
        mk.pose.orientation.w = math.cos(half)
        mk.scale.x = CLOCK_LENGTH_M
        mk.scale.y = CLOCK_WIDTH_M
        mk.scale.z = CLOCK_HEIGHT_M
        mk.color.r, mk.color.g, mk.color.b, mk.color.a = CLOCK_COLOR
        mk.lifetime = Duration(seconds=0.5).to_msg()
        self.marker_pub.publish(mk)

    def _publish_grasp_corner_marker(self, stamp_msg, grasp_target: list):
        """Publish an orange SPHERE at the grasp corner (same position the supervisor will target)."""
        mk = Marker()
        mk.header.frame_id = self.base_frame
        mk.header.stamp = stamp_msg
        mk.ns = "grasp_target"
        mk.id = 0
        mk.type = Marker.SPHERE
        mk.action = Marker.ADD
        mk.pose.position.x = float(grasp_target[0])
        mk.pose.position.y = float(grasp_target[1])
        mk.pose.position.z = float(grasp_target[2])
        mk.pose.orientation.w = 1.0
        mk.scale.x = 0.025
        mk.scale.y = 0.025
        mk.scale.z = 0.025
        mk.color.r = 1.0
        mk.color.g = 0.2
        mk.color.b = 0.0
        mk.color.a = 1.0
        mk.lifetime = Duration(seconds=0.5).to_msg()
        self.grasp_corner_pub.publish(mk)

    # ── Main processing callback ──
    def _on_rgb_depth(self, rgb_msg: Image, depth_msg: Image):
        now = time.time()
        if (now - self.last_t) < self.period:
            return
        self.last_t = now

        if self.fx is None:
            return

        try:
            bgr = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        except Exception:
            return
        try:
            depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except Exception:
            return

        depth_m = depth_to_meters(depth_raw, getattr(depth_msg, "encoding", ""))

        debug_image = bgr.copy()
        stamp_msg = self.get_clock().now().to_msg()
        h_img, w_img = depth_m.shape[:2]

        # ── Roboflow detection ──
        preds = self.rf.submit_and_get_latest(bgr)

        # Quality defaults
        clock_conf = 0.0
        pose_mode = 0.0
        tf_ok = 0.0
        valid_ratio = 0.0
        bbox_area = 0.0
        depth_med = float("nan")

        best = None
        best_conf = 0.0
        best_any = None
        best_any_conf = 0.0

        for pred in preds:
            cls_name = str(pred.get("cls", "")).strip().lower()
            conf = float(pred.get("conf", 0.0))
            if conf < self.rf_conf_threshold:
                continue
            if conf > best_any_conf:
                best_any_conf = conf
                best_any = pred
            if self.roboflow_class_name and cls_name != self.roboflow_class_name:
                continue
            if conf > best_conf:
                best_conf = conf
                best = pred

        if best is None:
            best = best_any
            best_conf = best_any_conf

        got_detection = False

        if best is not None:
            clock_conf = best_conf
            x1, y1, x2, y2, cu, cv_px = bbox_from_pred(best)
            bb = clip_bbox(x1, y1, x2, y2, w_img, h_img)

            if bb is not None:
                x1c, y1c, x2c, y2c = bb
                bbox_area = float((x2c - x1c) * (y2c - y1c))

                # Draw bbox
                label = str(best.get("cls", "clock"))
                cv2.rectangle(debug_image, (x1c, y1c), (x2c, y2c), (0, 255, 0), 2)
                cv2.putText(debug_image, f"{label} {best_conf:.2f}",
                            (x1c, max(0, y1c - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.circle(debug_image, (cu, cv_px), 4, (0, 0, 255), -1)

                # ── Depth sampling — inner bbox, near-surface cluster ──
                # Inner 70% of bbox reduces background contamination from edges
                _mx = max(1, int((x2c - x1c) * 0.15))
                _my = max(1, int((y2c - y1c) * 0.15))
                roi_depth = depth_m[y1c + _my : y2c - _my, x1c + _mx : x2c - _mx]
                vals_all = roi_depth[np.isfinite(roi_depth)]
                vals_all = vals_all[(vals_all > self.min_depth_m) & (vals_all <= self.max_depth_m)]

                if vals_all.size >= 20:
                    # Keep only points within 6 cm of the nearest surface — that is the
                    # clock's front face; pixels behind it are the shelf or background.
                    d_near = float(np.percentile(vals_all, 5))
                    vals = vals_all[vals_all <= d_near + 0.06]
                    if vals.size < 10:
                        vals = vals_all
                    d = float(np.median(vals))
                    depth_med = d
                    valid_ratio = float(vals.size) / max(1.0, float(roi_depth.size))

                    # Deproject bbox-pixel centroid (will be overridden below by 3D mean)
                    x_c = (cu - self.cx) * d / self.fx
                    y_c = (cv_px - self.cy) * d / self.fy
                    z_c = d

                    # TF lookup
                    tf_base_cam = self._lookup_base_T_cam()
                    if tf_base_cam is not None:
                        tf_ok = 1.0
                        result = self._transform_point(x_c, y_c, z_c, rgb_msg.header, tf_base_cam)
                        if result is not None:
                            wx, wy, wz = result

                            # Workspace check
                            if (self.ws_min_x <= wx <= self.ws_max_x and
                                    self.ws_min_y <= wy <= self.ws_max_y and
                                    self.ws_min_z <= wz <= self.ws_max_z):

                                # ── YAW EXTRACTION (PCA on inner-bbox, near-surface points) ──
                                # Use same inner margins as depth sampling to exclude
                                # background pixels at bbox edges.
                                xi1 = x1c + _mx; xi2 = x2c - _mx
                                yi1 = y1c + _my; yi2 = y2c - _my
                                binary = np.zeros(depth_m.shape[:2], dtype=np.uint8)
                                binary[yi1:yi2, xi1:xi2] = 1
                                vs_all, us_all = np.where(binary > 0)
                                us_s = us_all[::PCA_SUBSAMPLE_STEP]
                                vs_s = vs_all[::PCA_SUBSAMPLE_STEP]

                                pts_xy = []
                                pts_xyz_world = []
                                for u_px, v_px in zip(us_s, vs_s):
                                    d_px = float(depth_m[v_px, u_px])
                                    if d_px <= 0.0 or not np.isfinite(d_px):
                                        continue
                                    # Near-surface filter: same 6 cm band used above
                                    if not (self.min_depth_m <= d_px <= d + 0.06):
                                        continue
                                    xp = (u_px - self.cx) * d_px / self.fx
                                    yp = (v_px - self.cy) * d_px / self.fy
                                    zp = d_px
                                    pt = self._transform_point(
                                        xp, yp, zp, rgb_msg.header, tf_base_cam)
                                    if pt is not None:
                                        pts_xy.append([pt[0], pt[1]])
                                        pts_xyz_world.append([pt[0], pt[1], pt[2]])

                                yaw = 0.0
                                if len(pts_xy) >= PCA_MIN_POINTS:
                                    pts = np.array(pts_xy)
                                    mean_xy = pts.mean(axis=0)
                                    centered = pts - mean_xy
                                    cov = np.cov(centered.T)
                                    _, eigvecs = np.linalg.eigh(cov)
                                    long_axis = eigvecs[:, -1]
                                    yaw = float(math.atan2(long_axis[1], long_axis[0]))

                                    # Override centroid with 3D mean of surface points —
                                    # more accurate than deprojecting the 2D bbox centre.
                                    if len(pts_xyz_world) >= PCA_MIN_POINTS:
                                        _c3d = np.mean(pts_xyz_world, axis=0)
                                        wx = float(_c3d[0])
                                        wy = float(_c3d[1])
                                        wz = float(_c3d[2])

                                pose_mode = 1.0
                                got_detection = True

                                # ── EMA filter ──
                                new_xyz = np.array([wx, wy, wz], dtype=np.float64)
                                if not self.have_clock:
                                    self.clock_xyz = new_xyz
                                    self.clock_yaw = yaw
                                    self.have_clock = True
                                    self.get_logger().info(
                                        f"First clock detection: "
                                        f"({wx:.3f},{wy:.3f},{wz:.3f},y={yaw:.2f})")
                                else:
                                    jump = float(np.linalg.norm(new_xyz - self.clock_xyz))
                                    if jump > JUMP_RESET_M:
                                        self.clock_xyz = new_xyz
                                        self.clock_yaw = yaw
                                        self.get_logger().info(
                                            f"Clock jump reset ({jump:.3f}m)")
                                    elif jump <= MAX_JUMP_M:
                                        a = POS_ALPHA
                                        self.clock_xyz = a * new_xyz + (1 - a) * self.clock_xyz
                                        # Circular yaw smoothing
                                        dyaw = math.atan2(
                                            math.sin(yaw - self.clock_yaw),
                                            math.cos(yaw - self.clock_yaw))
                                        self.clock_yaw += YAW_ALPHA * dyaw

                                self.clock_miss_count = 0
                            else:
                                self._log_reject(
                                    f"workspace xyz=({wx:.3f},{wy:.3f},{wz:.3f}) "
                                    f"bounds=x[{self.ws_min_x:.2f},{self.ws_max_x:.2f}] "
                                    f"y[{self.ws_min_y:.2f},{self.ws_max_y:.2f}] "
                                    f"z[{self.ws_min_z:.2f},{self.ws_max_z:.2f}]"
                                )
                        else:
                            self._log_reject("transform point failed after TF lookup")
                else:
                    self._log_reject(
                        f"insufficient valid depth pixels in bbox "
                        f"(count={vals_all.size}, required>=20)"
                    )

        # ── Staleness ──
        if not got_detection:
            self.clock_miss_count += 1
            if self.clock_miss_count > STALE_FRAMES and self.have_clock:
                self.have_clock = False
                self.get_logger().info("[CLOCK] lost detection (stale)")

        # ── Publish ──
        if self.have_clock:
            ps = PoseStamped()
            ps.header.frame_id = self.base_frame
            ps.header.stamp = stamp_msg
            ps.pose.position.x = float(self.clock_xyz[0])
            ps.pose.position.y = float(self.clock_xyz[1])
            ps.pose.position.z = float(self.clock_xyz[2])
            # Encode yaw as quaternion (rotation about Z)
            half_yaw = float(self.clock_yaw) / 2.0
            ps.pose.orientation.x = 0.0
            ps.pose.orientation.y = 0.0
            ps.pose.orientation.z = math.sin(half_yaw)
            ps.pose.orientation.w = math.cos(half_yaw)
            self.pose_pub.publish(ps)

            self._publish_clock_marker(stamp_msg, self.clock_xyz, self.clock_yaw)

            # Compute and visualise the grasp corner (same formula as supervisor)
            _yaw = self.clock_yaw + self.yaw_offset_rad
            _lx = (CLOCK_LENGTH_M / 2.0) - self.grasp_end_inset_m
            _lz = CLOCK_HEIGHT_M / 2.0
            _cx = float(self.clock_xyz[0])
            _cy = float(self.clock_xyz[1])
            _cz = float(self.clock_xyz[2])
            _gt = [
                _cx - _lx * math.cos(_yaw),
                _cy - _lx * math.sin(_yaw),
                max(self.min_grasp_z_m, _cz + _lz),
            ]
            self._publish_grasp_corner_marker(stamp_msg, _gt)

            # Overlay on debug image
            cv2.putText(
                debug_image,
                f"({self.clock_xyz[0]:.3f},{self.clock_xyz[1]:.3f},"
                f"{self.clock_xyz[2]:.3f},y={self.clock_yaw:.2f})",
                (10, debug_image.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        # ── Quality topic ──
        q = Float32MultiArray()
        q.data = [
            float(clock_conf),
            float(pose_mode),
            float(tf_ok),
            float(valid_ratio),
            float(bbox_area),
            float(depth_med) if np.isfinite(depth_med) else 0.0,
        ]
        self.quality_pub.publish(q)

        # ── Quality CSV ──
        if self._enable_quality_csv and self._quality_csv_path:
            try:
                t_s = time.monotonic() - self._t0_mono
                row = [t_s, self._log_trial_id, self._log_variant] + [float(x) for x in q.data]
                with open(self._quality_csv_path, "a", newline="") as f:
                    csv.writer(f).writerow(row)
            except Exception:
                pass

        # ── Debug image ──
        try:
            msg = self.bridge.cv2_to_imgmsg(debug_image, encoding="bgr8")
            msg.header = rgb_msg.header
            self.debug_image_pub.publish(msg)
        except Exception:
            pass

        # ── Periodic log ──
        self._log_count += 1
        if (self._log_count % LOG_EVERY_N) == 0:
            self.get_logger().info(
                f"tf_ok={int(tf_ok)} conf={clock_conf:.2f} mode={int(pose_mode)} "
                f"depth={depth_med:.3f} vr={valid_ratio:.2f} "
                f"have={self.have_clock} "
                f"xyz=({self.clock_xyz[0]:.3f},{self.clock_xyz[1]:.3f},{self.clock_xyz[2]:.3f}) "
                f"yaw={self.clock_yaw:.2f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rf_api_key", default=ROBOFLOW_API_KEY_DEFAULT)
    parser.add_argument("--rf_project", default=ROBOFLOW_PROJECT_DEFAULT)
    parser.add_argument("--rf_version", default=ROBOFLOW_VERSION_DEFAULT)
    parser.add_argument("--rf_conf", type=float, default=ROBOFLOW_CONFIDENCE_DEFAULT)
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = ClockPoseNodeMetrics(args.rf_api_key, args.rf_project, args.rf_version, args.rf_conf)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.rf.shutdown()
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
