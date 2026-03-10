#!/usr/bin/env python3
"""
plant_pose_node_metrics.py  (ROS 2 Jazzy)

Adapted from fan_pose_buttons_node_metrics.py for the "Move Plant" task.

Detects a single class ("plants") via Roboflow, estimates its 3D position
in base_link using depth, and publishes:
    /plant/pose_base   PoseStamped   (position-only, identity orientation)
    /plant/quality     Float32MultiArray (layout documented below)
    /plant/markers     MarkerArray   (plant cube in RViz)

The supervisor subscribes to these topics to determine when the plant is
found and where it is, then executes the sweep motion.

Run:
  ros2 run bottle_grasping plant_pose_node_metrics --mode detect
"""

import os
import time
import csv
import base64
import threading
import numpy as np
import cv2
import requests

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge

import message_filters
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose_stamped


# ============================
# Roboflow config (IN SCRIPT)
# ============================
ROBOFLOW_API_KEY_DEFAULT = "" # Insert your Roboflow API key here
ROBOFLOW_PROJECT_DEFAULT = "plant-fjnqj"
ROBOFLOW_VERSION_DEFAULT = "1"
ROBOFLOW_CONFIDENCE_DEFAULT = 0.50

# Networking robustness
ROBOFLOW_TIMEOUT_S = 2.5
ROBOFLOW_RETRIES = 1


# ============================
# User-configurable parameters
# ============================

# Frames
BASE_FRAME = "base_link"
CAM_FRAME  = "camera_color_frame"

# Topics
RGB_TOPIC   = "/camera/color/image_raw"
DEPTH_TOPIC = "/camera/depth_registered/image_rect"
INFO_TOPIC  = "/camera/color/camera_info"

# Processing
PROCESS_HZ = 5.0

# TF robustness
TF_LOOKUP_TIMEOUT_S  = 0.25
TF_WARN_THROTTLE_S   = 1.0

# If multiple detections of same class:
TIEBREAK_BY_CLOSER_DEPTH = True
TIEBREAK_DEPTH_PATCH_PX = 9

# Depth gating
DEPTH_MIN_M = 0.10
DEPTH_MAX_M = 2.50
GATE_ABS_M  = 0.030
GATE_REL    = 0.05

# Minimum valid depth pixels to accept
MIN_MASK_PX_PLANT = 50

# Filtering / jump rejection (BASE frame)
MAX_JUMP_M_PLANT = 0.12
POS_ALPHA_PLANT  = 0.35

# Staleness: if not detected for N consecutive frames, reset to NaN
STALE_FRAMES_PLANT = 15    # 15 frames @ 5Hz = 3 seconds

# After a staleness reset, accumulate this many readings and publish
# the MEDIAN before switching to normal EMA.
CAPTURE_FRAMES_PLANT = 3   # 3 frames @ 5Hz = 0.6s burst

# If a detection jumps more than this, reset filter
JUMP_RESET_M_PLANT = 0.25  # >25cm = clearly new object or robot moved

# If detection stays in the dead zone for this many consecutive frames,
# accept it (object probably did move slightly)
DEADZONE_ACCEPT_COUNT = 5

# Unit conversion
INCH = 0.0254

# Marker sizes
# Pot dimensions: 3in x 3in x 5in
PLANT_MARKER_SIZE_X = 3.0 * INCH
PLANT_MARKER_SIZE_Y = 3.0 * INCH
PLANT_MARKER_SIZE_Z = 5.0 * INCH
# Target point offset: shift from visible/front surface toward pot center
# by half pot width in XY, then 1 inch lower in Z.
PLANT_TARGET_FORWARD_OFFSET_M = 0.6 * PLANT_MARKER_SIZE_X
# Target point offset: 1 inch below the plant center
PLANT_TARGET_Z_OFFSET_M = -1.0 * INCH
# Marker is rendered at the already-corrected target center.
PLANT_MARKER_FORWARD_OFFSET_M = 0.0
TEXT_SIZE_M = 0.045
TEXT_Z_OFFSET_M = 0.05

# Marker colors (RGBA)
PLANT_COLOR = (0.20, 0.85, 0.30, 0.55)  # green

MARKER_TOPIC  = "/plant/markers"
POSE_TOPIC    = "/plant/pose_base"
QUALITY_TOPIC = "/plant/quality"

# Classes (MATCH YOUR ROBOFLOW MODEL!)
CLS_PLANT = "plants"

# Logging
LOG_EVERY_N = 10
DEBUG_LOG_COUNTS = True

# Plant approximate physical size for smart depth sampling
PLANT_EXPECTED_SIZE_M = 0.15  # ~6 inches diameter estimate


# ============================
# Helpers
# ============================

def depth_to_meters(depth: np.ndarray, encoding: str = "") -> np.ndarray:
    """Convert depth image to float32 meters using the message encoding."""
    d = depth.astype(np.float32)
    enc_lower = encoding.lower()
    if enc_lower in ("16uc1", "mono16"):
        d /= 1000.0
    elif enc_lower in ("32fc1",):
        pass  # already meters
    elif depth.dtype in (np.uint16,):
        d /= 1000.0
    return d


def make_pose_stamped(frame_id, stamp_msg, p_xyz, q_xyzw):
    ps = PoseStamped()
    ps.header.frame_id = frame_id
    ps.header.stamp = stamp_msg
    ps.pose.position.x = float(p_xyz[0])
    ps.pose.position.y = float(p_xyz[1])
    ps.pose.position.z = float(p_xyz[2])
    ps.pose.orientation.x = float(q_xyzw[0])
    ps.pose.orientation.y = float(q_xyzw[1])
    ps.pose.orientation.z = float(q_xyzw[2])
    ps.pose.orientation.w = float(q_xyzw[3])
    return ps


def robust_depth_stats(depth_m, x1, y1, x2, y2, zmin, zmax):
    roi = depth_m[y1:y2, x1:x2]
    v = roi[np.isfinite(roi)]
    v = v[(v > zmin) & (v < zmax)]
    if v.size == 0:
        return None
    med = float(np.median(v))
    return med, float(np.mean(v)), float(np.std(v)), int(v.size), int(roi.size)


def backproject(u, v, Z, fx, fy, cx, cy):
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.array([X, Y, Z], dtype=np.float64)


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


def estimate_point_from_bbox_smart(depth_m, bbox, fx, fy, cx, cy,
                                   zmin, zmax,
                                   min_px=8,
                                   gate_abs=0.03, gate_rel=0.05,
                                   base_center_patch=9,
                                   expected_size_m=None):
    """Estimate the 3D point from a bounding box detection and depth image."""
    H, W = depth_m.shape[:2]
    x1, y1, x2, y2, uc, vc = bbox
    bb = clip_bbox(x1, y1, x2, y2, W, H)
    if bb is None:
        return None
    x1c, y1c, x2c, y2c = bb

    stats = robust_depth_stats(depth_m, x1c, y1c, x2c, y2c, zmin, zmax)
    if stats is None:
        return None
    z_med, _, _, n_valid, n_total = stats
    if n_valid < min_px:
        return None

    valid_ratio = float(n_valid) / float(max(1, n_total))
    dz = max(gate_abs, gate_rel * z_med)

    patch = base_center_patch
    if expected_size_m is not None:
        px_w = (expected_size_m * fx) / max(1e-6, z_med)
        patch = int(np.clip(round(px_w * 0.8), 7, 31))
        if patch % 2 == 0:
            patch += 1

    r = patch // 2
    u0 = int(np.clip(uc, 0, W - 1))
    v0 = int(np.clip(vc, 0, H - 1))
    x1p = max(0, u0 - r)
    x2p = min(W, u0 + r + 1)
    y1p = max(0, v0 - r)
    y2p = min(H, v0 + r + 1)

    pch = depth_m[y1p:y2p, x1p:x2p]
    pv = pch[np.isfinite(pch)]
    pv = pv[(pv > zmin) & (pv < zmax) & (np.abs(pv - z_med) < dz)]
    if pv.size < max(5, min_px // 2):
        roi = depth_m[y1c:y2c, x1c:x2c]
        v = roi[np.isfinite(roi)]
        v = v[(v > zmin) & (v < zmax) & (np.abs(v - z_med) < dz)]
        if v.size < min_px:
            return None
        z_use = float(np.median(v))
        used = int(v.size)
    else:
        z_use = float(np.median(pv))
        used = int(pv.size)

    p_cam = backproject(float(u0), float(v0), z_use, fx, fy, cx, cy)
    return p_cam, z_use, valid_ratio, int(n_valid), patch, used


# ============================
# Roboflow client (threaded)
# ============================

class RoboflowClient:
    """Threaded Roboflow client.  Inference runs in a background thread so the
    ROS callback never blocks on HTTP."""

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
            f"conf={self.conf:.2f}")

        # Threading state
        self._lock = threading.Lock()
        self._pending_bgr = None
        self._latest_result: list = []
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._stop = threading.Event()
        self._has_work = threading.Event()
        self._thread.start()

    def submit_and_get_latest(self, bgr: np.ndarray) -> list:
        """Non-blocking: enqueue *bgr* for the worker and return whatever
        results are already available."""
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
                        "cls": p.get("class", ""),
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

class PlantPoseNode(Node):
    def __init__(self, rf_api_key, rf_project, rf_version, rf_conf):
        super().__init__("plant_pose_node")
        self.rf_conf_threshold = float(rf_conf)

        # ── Optional CSV logging ──────────────────────────────────
        self.declare_parameter("trial_id", "")
        self.declare_parameter("variant", "")
        self.declare_parameter("log_dir", "")
        self.declare_parameter("enable_quality_csv", False)

        self._log_trial_id = str(self.get_parameter("trial_id").value)
        self._log_variant = str(self.get_parameter("variant").value)
        self._log_dir = str(self.get_parameter("log_dir").value)
        self._enable_quality_csv = bool(
            self.get_parameter("enable_quality_csv").value)

        self._t0_mono = time.monotonic()
        self._quality_csv_path = ""
        if self._enable_quality_csv and self._log_dir.strip():
            os.makedirs(self._log_dir, exist_ok=True)
            self._quality_csv_path = os.path.join(
                self._log_dir, "quality_stream.csv")
            if not os.path.exists(self._quality_csv_path):
                with open(self._quality_csv_path, "w", newline="") as f:
                    csv.writer(f).writerow([
                        "t_s", "trial_id", "variant",
                        "plant_conf", "pose_mode", "tf_ok",
                        "plant_valid_ratio", "plant_bbox_area_px",
                        "plant_depth_med", "plant_used_px"
                    ])

        self.bridge = CvBridge()
        self.rf = RoboflowClient(
            self.get_logger(), rf_api_key, rf_project, rf_version, rf_conf)

        # Intrinsics
        self.fx = self.fy = self.cx = self.cy = None
        self.create_subscription(CameraInfo, INFO_TOPIC, self.info_cb, 10)

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._last_tf_warn = 0.0

        # Publishers
        self.pose_pub = self.create_publisher(PoseStamped, POSE_TOPIC, 10)
        self.q_pub    = self.create_publisher(Float32MultiArray, QUALITY_TOPIC, 10)
        self.mk_pub   = self.create_publisher(MarkerArray, MARKER_TOPIC, 10)

        # Sync subs
        rgb_sub = message_filters.Subscriber(self, Image, RGB_TOPIC)
        d_sub   = message_filters.Subscriber(self, Image, DEPTH_TOPIC)
        sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, d_sub], queue_size=10, slop=0.08)
        sync.registerCallback(self.cb)

        # Rate limit
        self.last_t = 0.0
        self.period = 1.0 / max(1e-6, PROCESS_HZ)

        # Filters
        self.have_plant = False
        self.plant_p = np.zeros(3, dtype=np.float64)
        self.plant_miss_count = 0
        self.plant_deadzone_count = 0
        self.plant_capture_buf: list = []

        self._log_count = 0
        self.get_logger().info("PlantPoseNode running.")

    def info_cb(self, msg: CameraInfo):
        if self.fx is None:
            self.fx, self.fy = float(msg.k[0]), float(msg.k[4])
            self.cx, self.cy = float(msg.k[2]), float(msg.k[5])
            self.get_logger().info(
                f"Intrinsics: fx={self.fx:.2f} fy={self.fy:.2f} "
                f"cx={self.cx:.2f} cy={self.cy:.2f}")

    def _lookup_base_T_cam(self):
        now_s = time.time()
        try:
            tf = self.tf_buffer.lookup_transform(
                BASE_FRAME, CAM_FRAME, rclpy.time.Time(),
                timeout=Duration(seconds=TF_LOOKUP_TIMEOUT_S))
            return tf
        except Exception as e:
            if (now_s - self._last_tf_warn) > TF_WARN_THROTTLE_S:
                self.get_logger().warn(
                    f"TF lookup failed ({BASE_FRAME} <- {CAM_FRAME}): {e}")
                self._last_tf_warn = now_s
            return None

    def _transform_pose(self, ps_cam: PoseStamped, tf_base_cam):
        try:
            return do_transform_pose_stamped(ps_cam, tf_base_cam)
        except Exception as e:
            self.get_logger().warn(f"do_transform_pose_stamped failed: {e}")
            return None

    def _make_marker(self, stamp_msg, ns, mid, mtype, pos_xyz, quat_xyzw,
                     scale_xyz, color_rgba, text=""):
        mk = Marker()
        mk.header.frame_id = BASE_FRAME
        mk.header.stamp = stamp_msg
        mk.ns = ns
        mk.id = int(mid)
        mk.type = int(mtype)
        mk.action = Marker.ADD
        mk.pose.position.x = float(pos_xyz[0])
        mk.pose.position.y = float(pos_xyz[1])
        mk.pose.position.z = float(pos_xyz[2])
        mk.pose.orientation.x = float(quat_xyzw[0])
        mk.pose.orientation.y = float(quat_xyzw[1])
        mk.pose.orientation.z = float(quat_xyzw[2])
        mk.pose.orientation.w = float(quat_xyzw[3])
        mk.scale.x = float(scale_xyz[0])
        mk.scale.y = float(scale_xyz[1])
        mk.scale.z = float(scale_xyz[2])
        mk.color.r = float(color_rgba[0])
        mk.color.g = float(color_rgba[1])
        mk.color.b = float(color_rgba[2])
        mk.color.a = float(color_rgba[3])
        mk.lifetime = Duration(seconds=0.5).to_msg()
        if mtype == Marker.TEXT_VIEW_FACING:
            mk.text = text
        return mk

    def _publish_markers(self, stamp_msg, plant_pose_base: PoseStamped):
        arr = MarkerArray()
        if plant_pose_base is not None:
            p = plant_pose_base.pose.position
            q = plant_pose_base.pose.orientation
            # Marker-only center adjustment: move from front surface estimate
            # toward object center along base->plant XY direction.
            fxy = np.array([p.x, p.y], dtype=np.float64)
            nxy = float(np.linalg.norm(fxy))
            if nxy > 1e-6:
                ox = PLANT_MARKER_FORWARD_OFFSET_M * (fxy[0] / nxy)
                oy = PLANT_MARKER_FORWARD_OFFSET_M * (fxy[1] / nxy)
            else:
                ox = 0.0
                oy = 0.0
            mx = float(p.x + ox)
            my = float(p.y + oy)
            mz = float(p.z)
            arr.markers.append(
                self._make_marker(
                    stamp_msg,
                    ns="plant",
                    mid=0,
                    mtype=Marker.CUBE,
                    pos_xyz=(mx, my, mz),
                    quat_xyzw=(q.x, q.y, q.z, q.w),
                    scale_xyz=(PLANT_MARKER_SIZE_X,
                               PLANT_MARKER_SIZE_Y,
                               PLANT_MARKER_SIZE_Z),
                    color_rgba=PLANT_COLOR,
                )
            )
            arr.markers.append(
                self._make_marker(
                    stamp_msg,
                    ns="plant_txt",
                    mid=1,
                    mtype=Marker.TEXT_VIEW_FACING,
                    pos_xyz=(mx, my, mz + TEXT_Z_OFFSET_M),
                    quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                    scale_xyz=(0.0, 0.0, TEXT_SIZE_M),
                    color_rgba=PLANT_COLOR,
                    text="plants",
                )
            )
        if arr.markers:
            self.mk_pub.publish(arr)

    def _pick_best_plant(self, preds, depth_m):
        """Pick the best 'Plant' detection (highest confidence, tiebreak by
        closer depth)."""
        plants = [p for p in preds if p["cls"] == CLS_PLANT]
        if not plants:
            return None

        plants = sorted(plants, key=lambda x: x["conf"], reverse=True)
        if not TIEBREAK_BY_CLOSER_DEPTH or len(plants) == 1:
            return plants[0]

        best_conf = plants[0]["conf"]
        eps = 0.02
        candidates = [p for p in plants if (best_conf - p["conf"]) <= eps]

        H, W = depth_m.shape[:2]

        def center_depth(p):
            x1, y1, x2, y2, uc, vc = bbox_from_pred(p)
            r = max(1, TIEBREAK_DEPTH_PATCH_PX // 2)
            u0 = int(np.clip(uc, 0, W - 1))
            v0 = int(np.clip(vc, 0, H - 1))
            x1p = max(0, u0 - r)
            x2p = min(W, u0 + r + 1)
            y1p = max(0, v0 - r)
            y2p = min(H, v0 + r + 1)
            patch = depth_m[y1p:y2p, x1p:x2p]
            v = patch[np.isfinite(patch)]
            v = v[(v > DEPTH_MIN_M) & (v < DEPTH_MAX_M)]
            if v.size == 0:
                return float("inf")
            return float(np.median(v))

        candidates = sorted(candidates, key=center_depth)
        return candidates[0]

    def cb(self, rgb_msg: Image, depth_msg: Image):
        if self.fx is None:
            return

        now = time.time()
        if now - self.last_t < self.period:
            return
        self.last_t = now

        bgr = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        depth_m = depth_to_meters(depth_raw, encoding=depth_msg.encoding)

        preds = self.rf.submit_and_get_latest(bgr)

        if DEBUG_LOG_COUNTS and (self._log_count % LOG_EVERY_N) == 0:
            counts = {}
            for p in preds:
                counts[p["cls"]] = counts.get(p["cls"], 0) + 1
            self.get_logger().info(f"RF counts: {counts}")

        plant_pred = self._pick_best_plant(preds, depth_m)

        stamp_msg = self.get_clock().now().to_msg()

        # TF (may be missing temporarily)
        tf_base_cam = self._lookup_base_T_cam()
        base_ready = (tf_base_cam is not None)

        # Defaults for publishing
        plant_pose_base = None
        plant_conf = float(plant_pred["conf"]) if plant_pred is not None else 0.0
        plant_valid_ratio = 0.0
        plant_bbox_area = 0.0
        plant_depth_med = float("nan")
        plant_used_px = 0
        pose_mode = 0.0   # 1=position, 0=none

        # ─────────────────────────────────────────────────
        # Plant 3D position estimation
        # ─────────────────────────────────────────────────
        plant_detected_this_frame = False
        if (base_ready and plant_pred is not None
                and plant_pred["conf"] >= self.rf_conf_threshold):
            x1, y1, x2, y2, uc, vc = bbox_from_pred(plant_pred)
            H, W = depth_m.shape[:2]
            bb = clip_bbox(x1, y1, x2, y2, W, H)
            if bb is not None:
                x1c, y1c, x2c, y2c = bb
                plant_bbox_area = float((x2c - x1c) * (y2c - y1c))

                est = estimate_point_from_bbox_smart(
                    depth_m, (x1, y1, x2, y2, uc, vc),
                    self.fx, self.fy, self.cx, self.cy,
                    DEPTH_MIN_M, DEPTH_MAX_M,
                    min_px=MIN_MASK_PX_PLANT,
                    gate_abs=GATE_ABS_M, gate_rel=GATE_REL,
                    base_center_patch=15,
                    expected_size_m=PLANT_EXPECTED_SIZE_M
                )
                if est is not None:
                    p_cam, z_use, vr, n_valid, patch, used = est
                    plant_valid_ratio = vr
                    plant_depth_med = float(z_use)
                    plant_used_px = int(used)

                    ps_cam = make_pose_stamped(
                        CAM_FRAME, stamp_msg, p_cam,
                        [0.0, 0.0, 0.0, 1.0])
                    pw = self._transform_pose(ps_cam, tf_base_cam)
                    if pw is not None:
                        p_new = np.array([
                            pw.pose.position.x,
                            pw.pose.position.y,
                            pw.pose.position.z
                        ], dtype=np.float64)

                        plant_detected_this_frame = True
                        if self.have_plant:
                            # Still in capture burst?
                            if len(self.plant_capture_buf) < CAPTURE_FRAMES_PLANT:
                                self.plant_capture_buf.append(p_new.copy())
                                self.plant_p = np.median(
                                    self.plant_capture_buf, axis=0)
                            else:
                                # Normal EMA mode
                                jump = np.linalg.norm(p_new - self.plant_p)
                                if jump <= MAX_JUMP_M_PLANT:
                                    self.plant_p = (
                                        POS_ALPHA_PLANT * p_new
                                        + (1.0 - POS_ALPHA_PLANT) * self.plant_p)
                                    self.plant_deadzone_count = 0
                                elif jump <= JUMP_RESET_M_PLANT:
                                    self.plant_deadzone_count += 1
                                    if self.plant_deadzone_count >= DEADZONE_ACCEPT_COUNT:
                                        self.get_logger().info(
                                            f"Plant dead-zone {jump:.3f}m persisted "
                                            f"{self.plant_deadzone_count}x — accepting")
                                        self.plant_p = p_new
                                        self.plant_deadzone_count = 0
                                else:
                                    self.get_logger().info(
                                        f"Plant jump {jump:.3f}m > "
                                        f"{JUMP_RESET_M_PLANT}m — resetting")
                                    self.plant_p = p_new
                                    self.plant_capture_buf = [p_new.copy()]
                                    self.plant_deadzone_count = 0
                        else:
                            # Fresh start after staleness reset
                            self.plant_capture_buf = [p_new.copy()]
                            self.plant_p = p_new
                            self.have_plant = True
                            self.plant_deadzone_count = 0

                        p_target = self.plant_p.copy()
                        # Shift target from front-surface depth hit toward
                        # pot center along base->plant XY direction.
                        fxy = np.array([self.plant_p[0], self.plant_p[1]],
                                       dtype=np.float64)
                        nxy = float(np.linalg.norm(fxy))
                        if nxy > 1e-6:
                            p_target[0] += (
                                PLANT_TARGET_FORWARD_OFFSET_M * (fxy[0] / nxy))
                            p_target[1] += (
                                PLANT_TARGET_FORWARD_OFFSET_M * (fxy[1] / nxy))
                        p_target[2] += PLANT_TARGET_Z_OFFSET_M
                        plant_pose_base = make_pose_stamped(
                            BASE_FRAME, stamp_msg, p_target,
                            [0.0, 0.0, 0.0, 1.0])
                        pose_mode = 1.0

        # Staleness
        if plant_detected_this_frame:
            self.plant_miss_count = 0
        else:
            self.plant_miss_count += 1
            if self.have_plant and self.plant_miss_count >= STALE_FRAMES_PLANT:
                self.get_logger().info(
                    f"Plant stale ({self.plant_miss_count} misses) — "
                    f"resetting to NaN")
                self.have_plant = False
                self.plant_p = np.zeros(3, dtype=np.float64)
                self.plant_capture_buf = []

        # ─────────────────────────────────────────────────
        # Publish
        # ─────────────────────────────────────────────────
        if plant_pose_base is not None:
            self.pose_pub.publish(plant_pose_base)

        # /plant/quality layout:
        # [0] plant_conf
        # [1] pose_mode  (1=position, 0=none)
        # [2] tf_ok
        # [3] plant_valid_ratio
        # [4] plant_bbox_area_px
        # [5] plant_depth_median_m
        # [6] plant_used_px
        q = Float32MultiArray()
        q.data = [
            float(plant_conf),
            float(pose_mode),
            1.0 if base_ready else 0.0,
            float(plant_valid_ratio),
            float(plant_bbox_area),
            float(plant_depth_med),
            float(plant_used_px),
        ]
        self.q_pub.publish(q)

        if self._enable_quality_csv and self._quality_csv_path:
            try:
                t_s = time.monotonic() - self._t0_mono
                row = ([t_s, self._log_trial_id, self._log_variant]
                       + [float(x) for x in q.data])
                with open(self._quality_csv_path, "a", newline="") as f:
                    csv.writer(f).writerow(row)
            except Exception:
                pass

        if base_ready and plant_pose_base is not None:
            self._publish_markers(stamp_msg, plant_pose_base)

        # Log
        self._log_count += 1
        if (self._log_count % LOG_EVERY_N) == 0:
            self.get_logger().info(
                f"tf_ok={base_ready} plant_conf={plant_conf:.2f} "
                f"mode={int(pose_mode)} depth={plant_depth_med:.3f} "
                f"used_px={plant_used_px} vr={plant_valid_ratio:.2f} "
                f"have_plant={self.have_plant}"
            )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rf_api_key", default=ROBOFLOW_API_KEY_DEFAULT)
    parser.add_argument("--rf_project", default=ROBOFLOW_PROJECT_DEFAULT)
    parser.add_argument("--rf_version", default=ROBOFLOW_VERSION_DEFAULT)
    parser.add_argument("--rf_conf", type=float,
                        default=ROBOFLOW_CONFIDENCE_DEFAULT)
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = PlantPoseNode(
        args.rf_api_key, args.rf_project, args.rf_version, args.rf_conf)
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
