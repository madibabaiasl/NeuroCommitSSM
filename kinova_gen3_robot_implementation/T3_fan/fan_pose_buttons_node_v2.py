#!/usr/bin/env python3
"""
fan_pose_buttons_node.py  (ROS 2 Jazzy)

Updated:
- Adds HAC-style CV feasibility gating (stable(f_cv) for dwell + hysteresis)
- Publishes:
    /fan/cv_feasible   std_msgs/Bool
    /fan/cv_state      std_msgs/UInt8   (0=INFEAS, 1=BUILDING, 2=FEAS)
    /fan/cv_score      std_msgs/Float32 (debug score 0..1)
"""

import time
import base64
import numpy as np
import cv2
import requests

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Float32MultiArray, Bool, UInt8, Float32
from cv_bridge import CvBridge

import message_filters
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose_stamped


# ============================
# Roboflow config (IN SCRIPT)
# ============================
ROBOFLOW_API_KEY_DEFAULT = ""  # Insert your Roboflow API key here
ROBOFLOW_PROJECT_DEFAULT = "fan-buttons"
ROBOFLOW_VERSION_DEFAULT = "1"
ROBOFLOW_CONFIDENCE_DEFAULT = 0.50

ROBOFLOW_TIMEOUT_S = 2.5
ROBOFLOW_RETRIES = 1


# ============================
# User-configurable parameters
# ============================

BASE_FRAME = "base_link"
CAM_FRAME  = "camera_color_frame"

RGB_TOPIC   = "/camera/color/image_raw"
DEPTH_TOPIC = "/camera/depth_registered/image_rect"
INFO_TOPIC  = "/camera/color/camera_info"

PROCESS_HZ = 5.0

TF_LOOKUP_TIMEOUT_S  = 0.25
TF_WARN_THROTTLE_S   = 1.0

TIEBREAK_BY_CLOSER_DEPTH = True
TIEBREAK_DEPTH_PATCH_PX = 9

DEPTH_MIN_M = 0.10
DEPTH_MAX_M = 2.50
GATE_ABS_M  = 0.030
GATE_REL    = 0.05

MIN_MASK_PX_FAN = 900
MIN_MASK_PX_BTN = 8

FAN_BBOX_SHRINK = 0.80

RANSAC_ITERS = 140
PLANE_INLIER_THR_M = 0.006
MIN_INLIERS = 900

MAX_JUMP_M_FAN = 0.12
POS_ALPHA_FAN  = 0.35
ORI_ALPHA_FAN  = 0.35

MAX_JUMP_M_BTN = 0.10
POS_ALPHA_BTN  = 0.45

FAN_MARKER_SIZE_X = 0.28
FAN_MARKER_SIZE_Y = 0.28
FAN_MARKER_SIZE_Z = 0.35
BTN_SPHERE_DIAM_M = 0.025
TEXT_SIZE_M = 0.045
TEXT_Z_OFFSET_M = 0.05

FAN_COLOR   = (0.15, 0.80, 0.95, 0.45)
WHITE_COLOR = (1.00, 1.00, 1.00, 0.95)
GRAY_COLOR  = (0.65, 0.65, 0.65, 0.95)

MARKER_TOPIC = "/fan/markers"
POSE_TOPIC   = "/fan/pose_base"
BUTTONS_TOPIC = "/fan/buttons"
QUALITY_TOPIC = "/fan/quality"

# NEW topics
CV_FEAS_TOPIC  = "/fan/cv_feasible"
CV_STATE_TOPIC = "/fan/cv_state"
CV_SCORE_TOPIC = "/fan/cv_score"

CLS_FAN   = "fan"
CLS_WHITE = "white-button"
CLS_GRAY  = "gray-button"

ALLOW_FAN_POSITION_ONLY_FALLBACK = True

LOG_EVERY_N = 10
DEBUG_LOG_COUNTS = True


# ============================
# Button physical size
# ============================
INCH = 0.0254
BTN_W_M = 1.0 * INCH


# ============================
# CV Feasibility gate parameters
# ============================
# Raw thresholds
FAN_CONF_MIN_FOR_FEAS = 0.55
BTN_CONF_MIN_FOR_FEAS = 0.70
BTN_VALIDR_MIN_FOR_FEAS = 0.18
PLANE_RMS_MAX_FOR_FEAS = 0.012   # meters

# Build a score in [0,1]
SCORE_ON  = 0.85   # must be >= this for dwell to turn ON
SCORE_OFF = 0.70   # drop below this -> turn OFF (hysteresis)

DWELL_S = 0.60     # must stay >= SCORE_ON for this long to become feasible


# ============================
# Helpers
# ============================

def depth_to_meters(depth: np.ndarray) -> np.ndarray:
    d = depth.astype(np.float32)
    v = d[np.isfinite(d)]
    if v.size == 0:
        return d
    med = float(np.median(v))
    if med > 100.0:  # mm -> m
        d /= 1000.0
    return d

def quat_normalize(q):
    q = np.array(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n

def quat_slerp(q0, q1, t):
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        return quat_normalize(q0 + t * (q1 - q0))
    theta0 = np.arccos(dot)
    sin0 = np.sin(theta0)
    theta = theta0 * t
    s0 = np.sin(theta0 - theta) / sin0
    s1 = np.sin(theta) / sin0
    return quat_normalize(s0 * q0 + s1 * q1)

def quat_from_R(R: np.ndarray) -> np.ndarray:
    t = np.trace(R)
    if t > 0.0:
        S = np.sqrt(t + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
    return quat_normalize([qx, qy, qz, qw])

def plane_from_3pts(p0, p1, p2):
    v1 = p1 - p0
    v2 = p2 - p0
    n = np.cross(v1, v2)
    nn = np.linalg.norm(n)
    if nn < 1e-9:
        return None
    n = n / nn
    d = -np.dot(n, p0)
    return n, d

def ransac_plane(P: np.ndarray, iters=140, thr=0.006):
    N = P.shape[0]
    if N < 300:
        return None

    best = None
    best_count = 0
    idxs = np.arange(N)

    for _ in range(iters):
        i0, i1, i2 = np.random.choice(idxs, 3, replace=False)
        out = plane_from_3pts(P[i0], P[i1], P[i2])
        if out is None:
            continue
        n, d = out
        dist = np.abs(P @ n + d)
        inl = dist < thr
        c = int(inl.sum())
        if c > best_count:
            best_count = c
            best = (n, d, inl)

    if best is None:
        return None

    n0, d0, inl0 = best
    Pin = P[inl0]
    if Pin.shape[0] < 50:
        return None

    c = Pin.mean(axis=0)
    Q = Pin - c
    cov = (Q.T @ Q) / max(1, Q.shape[0] - 1)
    w, V = np.linalg.eigh(cov)
    n = V[:, np.argmin(w)]
    n = n / (np.linalg.norm(n) + 1e-12)
    d = -np.dot(n, c)

    dist = np.abs(P @ n + d)
    inl = dist < thr
    c_ref = P[inl].mean(axis=0) if inl.any() else c
    return n, d, inl, c_ref

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

def estimate_point_from_bbox_smart(depth_m, bbox, fx, fy, cx, cy,
                                   zmin, zmax,
                                   min_px=8,
                                   gate_abs=0.03, gate_rel=0.05,
                                   base_center_patch=9,
                                   expected_size_m=None):
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


def _finite(x: float) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False

def _clamp01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))

def _score01(val: float, lo: float, hi: float) -> float:
    if not _finite(val):
        return 0.0
    if hi <= lo:
        return 0.0
    return _clamp01((float(val) - lo) / (hi - lo))


# ============================
# Roboflow client
# ============================

class RoboflowClient:
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
        self.logger.info(f"[Roboflow] Using project={self.project} version={self.version} conf={self.conf:.2f}")

    def detect_all(self, bgr: np.ndarray):
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
                    self.logger.error(f"[Roboflow] HTTP {resp.status_code}: {resp.text[:200]}")
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


# ============================
# Main node
# ============================

class FanPoseButtonsNode(Node):
    def __init__(self, rf_api_key, rf_project, rf_version, rf_conf):
        super().__init__("fan_pose_buttons_node")
        self.bridge = CvBridge()

        self.rf = RoboflowClient(self.get_logger(), rf_api_key, rf_project, rf_version, rf_conf)

        self.fx = self.fy = self.cx = self.cy = None
        self.create_subscription(CameraInfo, INFO_TOPIC, self.info_cb, 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._last_tf_warn = 0.0

        self.pose_pub = self.create_publisher(PoseStamped, POSE_TOPIC, 10)
        self.btn_pub  = self.create_publisher(Float32MultiArray, BUTTONS_TOPIC, 10)
        self.q_pub    = self.create_publisher(Float32MultiArray, QUALITY_TOPIC, 10)
        self.mk_pub   = self.create_publisher(MarkerArray, MARKER_TOPIC, 10)

        # NEW pubs
        self.cv_feas_pub  = self.create_publisher(Bool, CV_FEAS_TOPIC, 10)
        self.cv_state_pub = self.create_publisher(UInt8, CV_STATE_TOPIC, 10)
        self.cv_score_pub = self.create_publisher(Float32, CV_SCORE_TOPIC, 10)

        # CV gate state
        self._cv_is_on = False
        self._cv_on_since = None  # wall time
        self._last_cv_score = 0.0

        rgb_sub = message_filters.Subscriber(self, Image, RGB_TOPIC)
        d_sub   = message_filters.Subscriber(self, Image, DEPTH_TOPIC)
        sync = message_filters.ApproximateTimeSynchronizer([rgb_sub, d_sub], queue_size=10, slop=0.08)
        sync.registerCallback(self.cb)

        self.last_t = 0.0
        self.period = 1.0 / max(1e-6, PROCESS_HZ)

        self.have_fan = False
        self.fan_p = np.zeros(3, dtype=np.float64)
        self.fan_q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

        self.have_white = False
        self.white_p = np.zeros(3, dtype=np.float64)

        self.have_gray = False
        self.gray_p = np.zeros(3, dtype=np.float64)

        self._log_count = 0
        self.get_logger().info("FanPoseButtonsNode running.")

    def info_cb(self, msg: CameraInfo):
        if self.fx is None:
            self.fx, self.fy = float(msg.k[0]), float(msg.k[4])
            self.cx, self.cy = float(msg.k[2]), float(msg.k[5])
            self.get_logger().info(
                f"Intrinsics: fx={self.fx:.2f} fy={self.fy:.2f} cx={self.cx:.2f} cy={self.cy:.2f}"
            )

    def _lookup_base_T_cam(self):
        now_s = time.time()
        try:
            tf = self.tf_buffer.lookup_transform(
                BASE_FRAME, CAM_FRAME, rclpy.time.Time(),
                timeout=Duration(seconds=TF_LOOKUP_TIMEOUT_S)
            )
            return tf
        except Exception as e:
            if (now_s - self._last_tf_warn) > TF_WARN_THROTTLE_S:
                self.get_logger().warn(f"TF lookup failed ({BASE_FRAME} <- {CAM_FRAME}): {e}")
                self._last_tf_warn = now_s
            return None

    def _transform_pose(self, ps_cam: PoseStamped, tf_base_cam):
        try:
            return do_transform_pose_stamped(ps_cam, tf_base_cam)
        except Exception as e:
            self.get_logger().warn(f"do_transform_pose_stamped failed: {e}")
            return None

    def _make_marker(self, stamp_msg, ns, mid, mtype, pos_xyz, quat_xyzw, scale_xyz, color_rgba, text=""):
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

    def _publish_markers(self, stamp_msg, fan_pose_base: PoseStamped,
                         white_p_base, gray_p_base,
                         white_ok, gray_ok):
        arr = MarkerArray()

        if fan_pose_base is not None:
            p = fan_pose_base.pose.position
            q = fan_pose_base.pose.orientation
            arr.markers.append(
                self._make_marker(
                    stamp_msg,
                    ns="fan",
                    mid=0,
                    mtype=Marker.CUBE,
                    pos_xyz=(p.x, p.y, p.z),
                    quat_xyzw=(q.x, q.y, q.z, q.w),
                    scale_xyz=(FAN_MARKER_SIZE_X, FAN_MARKER_SIZE_Y, FAN_MARKER_SIZE_Z),
                    color_rgba=FAN_COLOR,
                )
            )

        if white_ok:
            arr.markers.append(
                self._make_marker(
                    stamp_msg,
                    ns="fan_buttons",
                    mid=10,
                    mtype=Marker.SPHERE,
                    pos_xyz=(white_p_base[0], white_p_base[1], white_p_base[2]),
                    quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                    scale_xyz=(BTN_SPHERE_DIAM_M, BTN_SPHERE_DIAM_M, BTN_SPHERE_DIAM_M),
                    color_rgba=WHITE_COLOR,
                )
            )
            arr.markers.append(
                self._make_marker(
                    stamp_msg,
                    ns="fan_buttons_txt",
                    mid=11,
                    mtype=Marker.TEXT_VIEW_FACING,
                    pos_xyz=(white_p_base[0], white_p_base[1], white_p_base[2] + TEXT_Z_OFFSET_M),
                    quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                    scale_xyz=(0.0, 0.0, TEXT_SIZE_M),
                    color_rgba=WHITE_COLOR,
                    text="white-button",
                )
            )

        if gray_ok:
            arr.markers.append(
                self._make_marker(
                    stamp_msg,
                    ns="fan_buttons",
                    mid=20,
                    mtype=Marker.SPHERE,
                    pos_xyz=(gray_p_base[0], gray_p_base[1], gray_p_base[2]),
                    quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                    scale_xyz=(BTN_SPHERE_DIAM_M, BTN_SPHERE_DIAM_M, BTN_SPHERE_DIAM_M),
                    color_rgba=GRAY_COLOR,
                )
            )
            arr.markers.append(
                self._make_marker(
                    stamp_msg,
                    ns="fan_buttons_txt",
                    mid=21,
                    mtype=Marker.TEXT_VIEW_FACING,
                    pos_xyz=(gray_p_base[0], gray_p_base[1], gray_p_base[2] + TEXT_Z_OFFSET_M),
                    quat_xyzw=(0.0, 0.0, 0.0, 1.0),
                    scale_xyz=(0.0, 0.0, TEXT_SIZE_M),
                    color_rgba=GRAY_COLOR,
                    text="gray-button",
                )
            )

        if arr.markers:
            self.mk_pub.publish(arr)

    def _pick_best_per_class(self, preds, depth_m):
        by_cls = {CLS_FAN: [], CLS_WHITE: [], CLS_GRAY: []}
        for p in preds:
            c = p["cls"]
            if c in by_cls:
                by_cls[c].append(p)

        out = {}
        H, W = depth_m.shape[:2]

        for cls_name, plist in by_cls.items():
            if not plist:
                out[cls_name] = None
                continue

            plist = sorted(plist, key=lambda x: x["conf"], reverse=True)

            if not TIEBREAK_BY_CLOSER_DEPTH or len(plist) == 1:
                out[cls_name] = plist[0]
                continue

            best_conf = plist[0]["conf"]
            eps = 0.02
            candidates = [p for p in plist if (best_conf - p["conf"]) <= eps]

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
            out[cls_name] = candidates[0]

        return out

    def _update_and_publish_cv_gate(self, *,
                                   base_ready: bool,
                                   fan_conf: float,
                                   pose_mode: float,
                                   plane_rms: float,
                                   white_ok: bool,
                                   gray_ok: bool,
                                   white_conf: float,
                                   gray_conf: float,
                                   white_vr: float,
                                   gray_vr: float):
        # Hard feasibility (raw)
        raw_ok = bool(base_ready) and bool(white_ok) and bool(gray_ok)

        # Subscores
        s_fan = _score01(fan_conf, FAN_CONF_MIN_FOR_FEAS, 0.90)
        s_w   = _score01(white_conf, BTN_CONF_MIN_FOR_FEAS, 0.95)
        s_g   = _score01(gray_conf,  BTN_CONF_MIN_FOR_FEAS, 0.95)
        s_wvr = _score01(white_vr, BTN_VALIDR_MIN_FOR_FEAS, 0.60)
        s_gvr = _score01(gray_vr,  BTN_VALIDR_MIN_FOR_FEAS, 0.60)

        s_plane = 0.0
        if _finite(plane_rms):
            # smaller is better, 1 at 0, 0 at PLANE_RMS_MAX_FOR_FEAS
            s_plane = _clamp01((PLANE_RMS_MAX_FOR_FEAS - float(plane_rms)) / PLANE_RMS_MAX_FOR_FEAS)

        # Score: conservative, dominated by min button confidence and depth support
        score = 1.0
        score *= 1.0 if raw_ok else 0.0
        score *= min(s_w, s_g)
        score *= (0.5 + 0.5 * min(s_wvr, s_gvr))
        score *= (0.6 + 0.4 * min(s_fan, s_plane))
        score = float(np.clip(score, 0.0, 1.0))
        self._last_cv_score = score

        # Publish score
        self.cv_score_pub.publish(Float32(data=score))

        # Gate with dwell + hysteresis
        now_s = time.time()

        if not self._cv_is_on:
            if raw_ok and (score >= SCORE_ON):
                if self._cv_on_since is None:
                    self._cv_on_since = now_s
                elif (now_s - self._cv_on_since) >= DWELL_S:
                    self._cv_is_on = True
            else:
                self._cv_on_since = None
        else:
            if (not raw_ok) or (score < SCORE_OFF):
                self._cv_is_on = False
                self._cv_on_since = None

        # State enum
        # 0=INFEAS, 1=BUILDING, 2=FEAS
        state = 0
        if self._cv_is_on:
            state = 2
        elif raw_ok and score >= SCORE_ON:
            state = 1

        self.cv_feas_pub.publish(Bool(data=bool(self._cv_is_on)))
        self.cv_state_pub.publish(UInt8(data=int(state)))

    def cb(self, rgb_msg: Image, depth_msg: Image):
        if self.fx is None:
            return

        now = time.time()
        if now - self.last_t < self.period:
            return
        self.last_t = now

        bgr = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        depth_m = depth_to_meters(depth_raw)

        preds = self.rf.detect_all(bgr)
        if not preds:
            # still publish gate as infeasible
            self._update_and_publish_cv_gate(
                base_ready=False, fan_conf=0.0, pose_mode=0.0, plane_rms=float("nan"),
                white_ok=False, gray_ok=False, white_conf=0.0, gray_conf=0.0,
                white_vr=0.0, gray_vr=0.0
            )
            return

        if DEBUG_LOG_COUNTS and (self._log_count % LOG_EVERY_N) == 0:
            counts = {}
            for p in preds:
                counts[p["cls"]] = counts.get(p["cls"], 0) + 1
            self.get_logger().info(f"RF counts: {counts}")

        best = self._pick_best_per_class(preds, depth_m)
        fan_pred = best.get(CLS_FAN, None)
        w_pred   = best.get(CLS_WHITE, None)
        g_pred   = best.get(CLS_GRAY, None)

        stamp_msg = self.get_clock().now().to_msg()

        tf_base_cam = self._lookup_base_T_cam()
        base_ready = (tf_base_cam is not None)

        fan_pose_base = None
        fan_conf = float(fan_pred["conf"]) if fan_pred is not None else 0.0
        fan_valid_ratio = 0.0
        fan_bbox_area = 0.0
        fan_depth_med = float("nan")
        plane_rms = float("nan")
        inliers_count = 0
        pose_mode = 0.0

        # -------------------------
        # Fan 6DoF pose via plane
        # -------------------------
        if base_ready and (fan_pred is not None) and (fan_pred["conf"] >= ROBOFLOW_CONFIDENCE_DEFAULT):
            x1, y1, x2, y2, uc, vc = bbox_from_pred(fan_pred)
            H, W = depth_m.shape[:2]
            bb = clip_bbox(x1, y1, x2, y2, W, H)
            if bb is not None:
                x1c, y1c, x2c, y2c = bb
                fan_bbox_area = float((x2c - x1c) * (y2c - y1c))

                bw = (x2c - x1c)
                bh = (y2c - y1c)
                sx = int((1.0 - FAN_BBOX_SHRINK) * 0.5 * bw)
                sy = int((1.0 - FAN_BBOX_SHRINK) * 0.5 * bh)
                x1s, y1s = x1c + sx, y1c + sy
                x2s, y2s = x2c - sx, y2c - sy
                bb2 = clip_bbox(x1s, y1s, x2s, y2s, W, H)
                if bb2 is not None:
                    x1s, y1s, x2s, y2s = bb2

                    roi = depth_m[y1s:y2s, x1s:x2s]
                    v = roi[np.isfinite(roi)]
                    v = v[(v > DEPTH_MIN_M) & (v < DEPTH_MAX_M)]
                    if v.size >= MIN_MASK_PX_FAN:
                        fan_depth_med = float(np.median(v))
                        dz = max(GATE_ABS_M, GATE_REL * fan_depth_med)
                        mask = np.isfinite(roi) & (roi > DEPTH_MIN_M) & (roi < DEPTH_MAX_M) & (np.abs(roi - fan_depth_med) < dz)
                        fan_valid_ratio = float(mask.sum()) / float(max(1, roi.size))

                        if int(mask.sum()) >= MIN_MASK_PX_FAN:
                            vs, us = np.where(mask)
                            us = us.astype(np.float64) + x1s
                            vs = vs.astype(np.float64) + y1s
                            Zs = depth_m[vs.astype(int), us.astype(int)].astype(np.float64)

                            Xs = (us - self.cx) * Zs / self.fx
                            Ys = (vs - self.cy) * Zs / self.fy
                            P = np.stack([Xs, Ys, Zs], axis=1)

                            outp = ransac_plane(P, iters=RANSAC_ITERS, thr=PLANE_INLIER_THR_M)
                            if outp is not None:
                                n, d, inl, center_cam = outp
                                inliers_count = int(inl.sum())
                                if inliers_count >= MIN_INLIERS:
                                    Pin = P[inl]

                                    if np.dot(n, center_cam) > 0:
                                        n = -n

                                    Q = Pin - Pin.mean(axis=0, keepdims=True)
                                    Qp = Q - (Q @ n)[:, None] * n[None, :]
                                    cov = (Qp.T @ Qp) / max(1, Qp.shape[0] - 1)
                                    w_eval, V = np.linalg.eigh(cov)
                                    order = np.argsort(w_eval)[::-1]
                                    V = V[:, order]
                                    x_axis = V[:, 0]
                                    x_axis = x_axis - np.dot(x_axis, n) * n
                                    x_axis /= (np.linalg.norm(x_axis) + 1e-12)
                                    y_axis = np.cross(n, x_axis)
                                    y_axis /= (np.linalg.norm(y_axis) + 1e-12)

                                    R_cam = np.stack([x_axis, y_axis, n], axis=1)
                                    q_cam = quat_from_R(R_cam)

                                    dist = np.abs(P @ n + d)
                                    plane_rms = float(np.sqrt(np.mean((dist[inl]) ** 2)))

                                    ps_cam = make_pose_stamped(CAM_FRAME, stamp_msg, center_cam, q_cam)
                                    pw = self._transform_pose(ps_cam, tf_base_cam)
                                    if pw is not None:
                                        p_new = np.array([pw.pose.position.x, pw.pose.position.y, pw.pose.position.z], dtype=np.float64)
                                        q_new = np.array([pw.pose.orientation.x, pw.pose.orientation.y, pw.pose.orientation.z, pw.pose.orientation.w], dtype=np.float64)
                                        q_new = quat_normalize(q_new)

                                        if self.have_fan:
                                            if np.linalg.norm(p_new - self.fan_p) <= MAX_JUMP_M_FAN:
                                                self.fan_p = POS_ALPHA_FAN * p_new + (1.0 - POS_ALPHA_FAN) * self.fan_p
                                                self.fan_q = quat_slerp(self.fan_q, q_new, ORI_ALPHA_FAN)
                                        else:
                                            self.fan_p = p_new
                                            self.fan_q = q_new
                                            self.have_fan = True

                                        fan_pose_base = make_pose_stamped(BASE_FRAME, stamp_msg, self.fan_p, self.fan_q)
                                        pose_mode = 2.0

        # -------------------------
        # Fan fallback: position-only
        # -------------------------
        if base_ready and (fan_pose_base is None) and ALLOW_FAN_POSITION_ONLY_FALLBACK and (fan_pred is not None):
            x1, y1, x2, y2, uc, vc = bbox_from_pred(fan_pred)
            est = estimate_point_from_bbox_smart(
                depth_m, (x1, y1, x2, y2, uc, vc),
                self.fx, self.fy, self.cx, self.cy,
                DEPTH_MIN_M, DEPTH_MAX_M,
                min_px=MIN_MASK_PX_BTN,
                gate_abs=GATE_ABS_M, gate_rel=GATE_REL,
                base_center_patch=15,
                expected_size_m=0.20
            )
            if est is not None:
                p_cam, z_use, vr, n_valid, patch, used = est
                ps_cam = make_pose_stamped(CAM_FRAME, stamp_msg, p_cam, [0.0, 0.0, 0.0, 1.0])
                pw = self._transform_pose(ps_cam, tf_base_cam)
                if pw is not None:
                    p_new = np.array([pw.pose.position.x, pw.pose.position.y, pw.pose.position.z], dtype=np.float64)
                    if self.have_fan:
                        if np.linalg.norm(p_new - self.fan_p) <= MAX_JUMP_M_FAN:
                            self.fan_p = POS_ALPHA_FAN * p_new + (1.0 - POS_ALPHA_FAN) * self.fan_p
                    else:
                        self.fan_p = p_new
                        self.have_fan = True

                    fan_pose_base = make_pose_stamped(BASE_FRAME, stamp_msg, self.fan_p, [0.0, 0.0, 0.0, 1.0])
                    pose_mode = 1.0
                    fan_depth_med = float(z_use)
                    fan_valid_ratio = float(vr)

        # -------------------------
        # Button 3D points
        # -------------------------
        def get_btn_point(pred):
            if (not base_ready) or pred is None or pred["conf"] < ROBOFLOW_CONFIDENCE_DEFAULT:
                return (False, np.array([np.nan, np.nan, np.nan], dtype=np.float64),
                        float(pred["conf"]) if pred is not None else 0.0,
                        float("nan"), 0.0, 0, 0)

            x1, y1, x2, y2, uc, vc = bbox_from_pred(pred)
            est = estimate_point_from_bbox_smart(
                depth_m, (x1, y1, x2, y2, uc, vc),
                self.fx, self.fy, self.cx, self.cy,
                DEPTH_MIN_M, DEPTH_MAX_M,
                min_px=MIN_MASK_PX_BTN,
                gate_abs=GATE_ABS_M, gate_rel=GATE_REL,
                base_center_patch=9,
                expected_size_m=BTN_W_M
            )
            if est is None:
                return (False, np.array([np.nan, np.nan, np.nan], dtype=np.float64),
                        float(pred["conf"]), float("nan"), 0.0, 0, 0)

            p_cam, z_use, vr, n_valid, patch, used = est
            ps_cam = make_pose_stamped(CAM_FRAME, stamp_msg, p_cam, [0.0, 0.0, 0.0, 1.0])
            pw = self._transform_pose(ps_cam, tf_base_cam)
            if pw is None:
                return (False, np.array([np.nan, np.nan, np.nan], dtype=np.float64),
                        float(pred["conf"]), float(z_use), float(vr), int(patch), int(used))

            p_new = np.array([pw.pose.position.x, pw.pose.position.y, pw.pose.position.z], dtype=np.float64)
            return (True, p_new, float(pred["conf"]), float(z_use), float(vr), int(patch), int(used))

        w_ok, w_p, w_conf, w_z, w_vr, w_patch, w_used = get_btn_point(w_pred)
        g_ok, g_p, g_conf, g_z, g_vr, g_patch, g_used = get_btn_point(g_pred)

        if w_ok:
            if self.have_white:
                if np.linalg.norm(w_p - self.white_p) <= MAX_JUMP_M_BTN:
                    self.white_p = POS_ALPHA_BTN * w_p + (1.0 - POS_ALPHA_BTN) * self.white_p
            else:
                self.white_p = w_p
                self.have_white = True

        if g_ok:
            if self.have_gray:
                if np.linalg.norm(g_p - self.gray_p) <= MAX_JUMP_M_BTN:
                    self.gray_p = POS_ALPHA_BTN * g_p + (1.0 - POS_ALPHA_BTN) * self.gray_p
            else:
                self.gray_p = g_p
                self.have_gray = True

        w_pub_ok = self.have_white
        g_pub_ok = self.have_gray
        w_pub = self.white_p if self.have_white else np.array([np.nan, np.nan, np.nan], dtype=np.float64)
        g_pub = self.gray_p if self.have_gray else np.array([np.nan, np.nan, np.nan], dtype=np.float64)

        # -------------------------
        # Publish pose/buttons/quality
        # -------------------------
        if fan_pose_base is not None:
            self.pose_pub.publish(fan_pose_base)

        btn = Float32MultiArray()
        btn.data = [
            float(w_pub[0]), float(w_pub[1]), float(w_pub[2]),
            float(g_pub[0]), float(g_pub[1]), float(g_pub[2]),
        ]
        self.btn_pub.publish(btn)

        q = Float32MultiArray()
        q.data = [
            float(fan_conf),
            float(pose_mode),
            1.0 if base_ready else 0.0,
            float(fan_valid_ratio),
            float(fan_bbox_area),
            float(fan_depth_med),
            float(plane_rms),
            float(inliers_count),
            float(w_conf), float(w_z), float(w_vr), float(w_patch), float(w_used),
            float(g_conf), float(g_z), float(g_vr), float(g_patch), float(g_used),
        ]
        self.q_pub.publish(q)

        if base_ready:
            self._publish_markers(stamp_msg, fan_pose_base, w_pub, g_pub, w_pub_ok, g_pub_ok)

        # -------------------------
        # NEW: publish stable CV feasibility
        # -------------------------
        self._update_and_publish_cv_gate(
            base_ready=base_ready,
            fan_conf=fan_conf,
            pose_mode=pose_mode,
            plane_rms=plane_rms,
            white_ok=w_pub_ok,
            gray_ok=g_pub_ok,
            white_conf=w_conf,
            gray_conf=g_conf,
            white_vr=w_vr,
            gray_vr=g_vr,
        )

        # Log
        self._log_count += 1
        if (self._log_count % LOG_EVERY_N) == 0:
            rms_mm = plane_rms * 1000.0 if np.isfinite(plane_rms) else float("nan")
            self.get_logger().info(
                f"cv_feas={self._cv_is_on} score={self._last_cv_score:.2f} | "
                f"tf_ok={base_ready} fan_conf={fan_conf:.2f} mode={int(pose_mode)} depth={fan_depth_med:.3f} "
                f"inl={inliers_count} rms_mm={rms_mm:.2f} | "
                f"white_ok={w_pub_ok} (conf={w_conf:.2f},vr={w_vr:.2f},used={w_used}) "
                f"gray_ok={g_pub_ok} (conf={g_conf:.2f},vr={g_vr:.2f},used={g_used})"
            )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rf_api_key", default=ROBOFLOW_API_KEY_DEFAULT)
    parser.add_argument("--rf_project", default=ROBOFLOW_PROJECT_DEFAULT)
    parser.add_argument("--rf_version", default=ROBOFLOW_VERSION_DEFAULT)
    parser.add_argument("--rf_conf", type=float, default=ROBOFLOW_CONFIDENCE_DEFAULT)
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = FanPoseButtonsNode(args.rf_api_key, args.rf_project, args.rf_version, args.rf_conf)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
