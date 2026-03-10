#!/usr/bin/env python3
"""
fan_press_supervisor.py  (ROS 2 Jazzy)

Unified supervisor + motion executor for the fan-button-press task.
Merges the old fan_press_supervisor + fan_grasp_loop + fan_buttons_bridge
into ONE node so that CV monitoring works reliably during execution.
"""

import math
import time
import threading
from collections import deque
from typing import Optional, List, Tuple
from copy import deepcopy

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from moveit_msgs.msg import MoveItErrorCodes
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Trigger
from geometry_msgs.msg import Pose, PoseStamped, WrenchStamped
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import CollisionObject as CollisionObjectMsg

from moveit_configs_utils import MoveItConfigsBuilder
from moveit.planning import MoveItPy, PlanRequestParameters
from moveit.core.robot_state import RobotState


# =====================================================================
#  Constants
# =====================================================================

INCH_TO_M = 0.0254

# States
PRE_SCAN = "PRE_SCAN"
HOLD     = "HOLD"
ASSIST   = "ASSIST"
EXECUTE  = "EXECUTE"

STATE_ID  = {PRE_SCAN: 0, HOLD: 1, ASSIST: 2, EXECUTE: 3}
VARIANT_ID = {"commit_only": 0, "feas_only": 1, "hold_commit": 2, "hac": 3}

# Outcome codes (for /fan/sup_event)
OUTCOME_SUCCESS = 1
OUTCOME_ABORT   = 2
WAIT_FOR_VISION = "WAIT_FOR_VISION"
WAIT_FOR_OBSTACLE = "WAIT_FOR_OBSTACLE"

# Quality indices
Q_FAN_CONF  = 0;  Q_POSE_MODE = 1;  Q_TF_OK    = 2
Q_FAN_VR    = 3;  Q_FAN_BBOX  = 4;  Q_FAN_DEPTH = 5
Q_PLANE_RMS = 6;  Q_PLANE_INL = 7
Q_W_CONF    = 8;  Q_W_DEPTH   = 9;  Q_W_VR     = 10
Q_W_PATCH   = 11; Q_W_USED    = 12
Q_G_CONF    = 13; Q_G_DEPTH   = 14; Q_G_VR     = 15
Q_G_PATCH   = 16; Q_G_USED    = 17
Q_MIN_LEN   = 18

# Scan poses (joint-space, 6 joints, gripper excluded)
SCAN_POSES: List[List[float]] = [
    [0.6427, 0.9756, 1.4850, 0.2295, 1.2395, -1.6416],  # 2
    [0.5322, 0.8251, 1.2608, 0.1999, 1.6878, -1.6803],  # 3
    [1.5701, 0.7471, 1.3472, 0.1524, 1.2910, -1.6665],  # 4
    [1.2839, 0.8440, 1.3817, 0.2653, 1.4562, -1.6168],  # 1
]


# =====================================================================
#  Geometry helpers
# =====================================================================

def _normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Cannot normalize near-zero vector")
    return v / n

def _rotm_to_quat_xyzw(R: np.ndarray) -> Tuple[float, float, float, float]:
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2.0
        qw, qx = 0.25 * S, (R[2, 1] - R[1, 2]) / S
        qy, qz = (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw, qx = (R[2, 1] - R[1, 2]) / S, 0.25 * S
        qy, qz = (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw, qx = (R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S
        qy, qz = 0.25 * S, (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw, qx = (R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S
        qy, qz = (R[1, 2] + R[2, 1]) / S, 0.25 * S
    q = np.array([qx, qy, qz, qw], dtype=float)
    q /= np.linalg.norm(q)
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])

def _make_orientation_quat(
    forward_world: np.ndarray,
    up_world: np.ndarray,
    tool_forward_axis: str = "z",
    tool_up_axis: str = "y",
) -> Tuple[float, float, float, float]:
    f = _normalize(forward_world)
    u = _normalize(up_world - np.dot(up_world, f) * f)
    axes = {"x": None, "y": None, "z": None}
    axes[tool_forward_axis] = f
    axes[tool_up_axis] = u
    missing = [a for a in ("x", "y", "z") if axes[a] is None][0]
    if missing == "x":
        axes["x"] = _normalize(np.cross(axes["y"], axes["z"]))
        axes["y"] = _normalize(np.cross(axes["z"], axes["x"]))
    elif missing == "y":
        axes["y"] = _normalize(np.cross(axes["z"], axes["x"]))
        axes["x"] = _normalize(np.cross(axes["y"], axes["z"]))
    else:
        axes["z"] = _normalize(np.cross(axes["x"], axes["y"]))
        axes["y"] = _normalize(np.cross(axes["z"], axes["x"]))
    R = np.column_stack([axes["x"], axes["y"], axes["z"]])
    return _rotm_to_quat_xyzw(R)

def _compute_forward(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    z_hat = np.array([0.0, 0.0, 1.0])
    left = (p1 - p2).copy(); left[2] = 0.0; left = _normalize(left)
    fwd = np.cross(left, z_hat); fwd[2] = 0.0; fwd = _normalize(fwd)
    mid = 0.5 * (p1 + p2); mid_xy = np.array([mid[0], mid[1], 0.0])
    if np.linalg.norm(mid_xy) > 1e-6:
        if np.dot(fwd, _normalize(mid_xy)) < 0.0:
            fwd = -fwd
    return fwd

def _augment_cfg(cfg: dict, jst: str = "/joint_states") -> dict:
    cfg = deepcopy(cfg)
    pips = cfg.get("planning_pipelines", [])
    if isinstance(pips, dict):
        return cfg
    if not pips:
        pips = ["ompl"]
    default_p = cfg.get("default_planning_pipeline", pips[0])
    cfg["planning_pipelines"] = {"pipeline_names": pips}
    cfg.setdefault("planning_scene_monitor_options", {
        "name": "planning_scene_monitor",
        "robot_description": "robot_description",
        "joint_state_topic": jst,
        "attached_collision_object_topic": "/attached_collision_object",
        "publish_planning_scene_topic": "/moveit_py/publish_planning_scene",
        "monitored_planning_scene_topic": "/moveit_py/monitored_planning_scene",
        "wait_for_initial_state_timeout": 10.0,
    })
    cfg.setdefault("plan_request_params", {
        "planning_attempts": 1,
        "planning_pipeline": default_p,
        "max_velocity_scaling_factor": 0.3,
        "max_acceleration_scaling_factor": 0.3,
        "planning_time": 2.0,
    })
    for p in pips:
        if p in cfg and isinstance(cfg[p], dict):
            if "planning_plugins" not in cfg[p] and "planning_plugin" in cfg[p]:
                cfg[p]["planning_plugins"] = [cfg[p]["planning_plugin"]]
    cfg.setdefault("pilz_lin", {"plan_request_params": {
        "planning_attempts": 1,
        "planning_pipeline": "pilz_industrial_motion_planner",
        "planner_id": "LIN",
        "max_velocity_scaling_factor": 0.05,
        "max_acceleration_scaling_factor": 0.05,
        "planning_time": 2.0,
    }})
    return cfg


# =====================================================================
#  The node
# =====================================================================

class FanPressSupervisor(Node):
    def __init__(self):
        super().__init__("fan_press_supervisor")

        # ── Callback groups ──────────────────────────────────────────
        self._sub_cg = ReentrantCallbackGroup()
        self._srv_cg = MutuallyExclusiveCallbackGroup()

        # ── Variant ──────────────────────────────────────────────────
        self.declare_parameter("variant", "feas_only")
        self.variant = self.get_parameter("variant").value.strip()
        if self.variant not in VARIANT_ID:
            raise ValueError(f"Unknown variant '{self.variant}'")
        self.declare_parameter("commit_only_single_shot", True)
        self.commit_only_single_shot = bool(self.get_parameter("commit_only_single_shot").value)

        # ── Trial ────────────────────────────────────────────────────
        self.declare_parameter("trial_id", 0)
        self.trial_id = int(self.get_parameter("trial_id").value)
        self.trial_t0 = time.monotonic()

        # ── PRE_SCAN params ──────────────────────────────────────────
        self.declare_parameter("enable_prescan", True)
        self.declare_parameter("min_fan_conf", 0.50)
        self.declare_parameter("prescan_dwell_sec", 0.40)
        self.declare_parameter("prescan_pose_settle_sec", 2.2)
        self.declare_parameter("scan_extra_pause_sec", 0.6)
        self.declare_parameter("prescan_max_cycles", 1000000)
        self.declare_parameter("scan_joint2_limit_deg", 47.0)

        self.enable_prescan      = bool(self.get_parameter("enable_prescan").value)
        self.min_fan_conf        = float(self.get_parameter("min_fan_conf").value)
        self.prescan_dwell       = float(self.get_parameter("prescan_dwell_sec").value)
        self.prescan_settle      = float(self.get_parameter("prescan_pose_settle_sec").value)
        self.scan_extra_pause    = float(self.get_parameter("scan_extra_pause_sec").value)
        self.prescan_max_cycles  = int(self.get_parameter("prescan_max_cycles").value)
        self.scan_joint2_limit_deg = max(0.0, float(self.get_parameter("scan_joint2_limit_deg").value))
        self.scan_joint2_limit_rad = math.radians(self.scan_joint2_limit_deg)

        # ── Feasibility params ───────────────────────────────────────
        self.declare_parameter("min_button_conf", 0.50)
        self.declare_parameter("min_used_px", 10.0)
        self.declare_parameter("min_valid_ratio", 0.20)
        self.declare_parameter("dwell_sec", 0.60)
        self.declare_parameter("retain_last_buttons_on_nan", True)
        self.declare_parameter("button_stale_sec", 8.0)
        self.declare_parameter("abort_on_cv_loss_during_execute", False)
        self.declare_parameter("cv_abort_phase", "all")
        self.declare_parameter("feas_only_live_pair_timeout_sec", 0.35)
        self.declare_parameter("pause_on_vision_loss", False)
        self.declare_parameter("pause_vision_timeout_sec", 20.0)
        self.declare_parameter("resume_then_ignore_vision_loss", True)
        self.declare_parameter("hac_allow_latched_execute", True)
        self.declare_parameter("clear_buttons_on_assist", False)
        self.declare_parameter("button_pair_min_dist_m", 0.003)
        self.declare_parameter("button_pair_max_dist_m", 0.060)

        self.min_btn_conf    = float(self.get_parameter("min_button_conf").value)
        self.min_used_px     = float(self.get_parameter("min_used_px").value)
        self.min_valid_ratio = float(self.get_parameter("min_valid_ratio").value)
        self.dwell_sec       = float(self.get_parameter("dwell_sec").value)
        self.retain_last_buttons_on_nan = bool(self.get_parameter("retain_last_buttons_on_nan").value)
        self._btn_stale_sec = float(self.get_parameter("button_stale_sec").value)
        self.abort_on_cv_loss_during_execute = bool(self.get_parameter("abort_on_cv_loss_during_execute").value)
        self.cv_abort_phase = str(self.get_parameter("cv_abort_phase").value).strip().lower()
        if self.cv_abort_phase not in ("all", "press_only"):
            self.cv_abort_phase = "all"
        self.feas_only_live_pair_timeout_sec = max(0.05, float(self.get_parameter("feas_only_live_pair_timeout_sec").value))
        self.pause_on_vision_loss = bool(self.get_parameter("pause_on_vision_loss").value)
        self.pause_vision_timeout_sec = max(0.0, float(self.get_parameter("pause_vision_timeout_sec").value))
        self.resume_then_ignore_vision_loss = bool(self.get_parameter("resume_then_ignore_vision_loss").value)
        self.hac_allow_latched_execute = bool(self.get_parameter("hac_allow_latched_execute").value)
        self.clear_buttons_on_assist = bool(self.get_parameter("clear_buttons_on_assist").value)
        self.button_pair_min_dist_m = float(self.get_parameter("button_pair_min_dist_m").value)
        self.button_pair_max_dist_m = float(self.get_parameter("button_pair_max_dist_m").value)

        # ── Force/Torque Safety params ───────────────────────────────
        self.declare_parameter("force_topic", "/wrench")
        self.declare_parameter("max_force_n", 25.0)
        self.force_topic = str(self.get_parameter("force_topic").value)
        self.max_force_n = float(self.get_parameter("max_force_n").value)
        self._force_lock = threading.Lock()
        self._force_exceeded = False

        # ── Depth-based obstacle gate ────────────────────────────────
        self.declare_parameter("obstacle_monitor_enabled", True)
        self.declare_parameter("obstacle_depth_topic", "/camera/depth_registered/image_rect")
        self.declare_parameter("obstacle_info_topic", "/camera/color/camera_info")
        self.declare_parameter("obstacle_stop_distance_m", 0.20)
        self.declare_parameter("obstacle_resume_distance_m", 0.28)
        self.declare_parameter("obstacle_roi_fraction", 0.30)
        self.declare_parameter("obstacle_min_valid_pixels", 80)
        self.declare_parameter("obstacle_min_valid_ratio", 0.03)
        self.declare_parameter("obstacle_percentile", 5.0)
        self.declare_parameter("obstacle_median_window", 5)
        self.declare_parameter("obstacle_stop_confirm_frames", 3)
        self.declare_parameter("obstacle_resume_confirm_frames", 6)
        self.declare_parameter("obstacle_fail_safe_on_no_depth", True)
        self.declare_parameter("obstacle_depth_timeout_sec", 1.0)
        self.declare_parameter("obstacle_clear_hold_sec", 0.25)
        self.declare_parameter("obstacle_pause_timeout_sec", 20.0)

        self.obstacle_monitor_enabled = bool(self.get_parameter("obstacle_monitor_enabled").value)
        self.obstacle_depth_topic = str(self.get_parameter("obstacle_depth_topic").value)
        self.obstacle_info_topic = str(self.get_parameter("obstacle_info_topic").value)
        self.obstacle_stop_distance_m = max(0.01, float(self.get_parameter("obstacle_stop_distance_m").value))
        self.obstacle_resume_distance_m = float(self.get_parameter("obstacle_resume_distance_m").value)
        if self.obstacle_resume_distance_m <= self.obstacle_stop_distance_m:
            self.obstacle_resume_distance_m = self.obstacle_stop_distance_m + 0.08
        self.obstacle_roi_fraction = min(1.0, max(0.05, float(self.get_parameter("obstacle_roi_fraction").value)))
        self.obstacle_min_valid_pixels = max(1, int(self.get_parameter("obstacle_min_valid_pixels").value))
        self.obstacle_min_valid_ratio = min(1.0, max(0.0, float(self.get_parameter("obstacle_min_valid_ratio").value)))
        self.obstacle_percentile = min(50.0, max(0.1, float(self.get_parameter("obstacle_percentile").value)))
        self.obstacle_median_window = max(1, int(self.get_parameter("obstacle_median_window").value))
        self.obstacle_stop_confirm_frames = max(1, int(self.get_parameter("obstacle_stop_confirm_frames").value))
        self.obstacle_resume_confirm_frames = max(1, int(self.get_parameter("obstacle_resume_confirm_frames").value))
        self.obstacle_fail_safe_on_no_depth = bool(self.get_parameter("obstacle_fail_safe_on_no_depth").value)
        self.obstacle_depth_timeout_sec = max(0.0, float(self.get_parameter("obstacle_depth_timeout_sec").value))
        self.obstacle_clear_hold_sec = max(0.0, float(self.get_parameter("obstacle_clear_hold_sec").value))
        self.obstacle_pause_timeout_sec = max(0.0, float(self.get_parameter("obstacle_pause_timeout_sec").value))
        self._obstacle_gate_enabled = (self.obstacle_monitor_enabled and self.variant in ("feas_only", "hold_commit", "hac"))
        
        if self.variant == "feas_only":
            self.abort_on_cv_loss_during_execute = True
            self.cv_abort_phase = "press_only"
            self.hac_allow_latched_execute = False
            self.retain_last_buttons_on_nan = False
            self._btn_stale_sec = min(self._btn_stale_sec, self.feas_only_live_pair_timeout_sec)
            self.pause_on_vision_loss = True

        # ── ASSIST params (HAC) ──────────────────────────────────────
        self.declare_parameter("assist_pose_settle_sec", 2.2)
        self.declare_parameter("assist_max_cycles", 10)
        self.assist_settle = float(self.get_parameter("assist_pose_settle_sec").value)
        self.assist_max_cycles = int(self.get_parameter("assist_max_cycles").value)
        self.assist_cycles = 0

        # ── Press execution params ───────────────────────────────────
        self.declare_parameter("press_timeout_sec", 90.0)
        self.declare_parameter("cv_abort_grace_sec", 1.5)
        self._press_timeout_sec = float(self.get_parameter("press_timeout_sec").value)
        self._cv_abort_grace_sec = float(self.get_parameter("cv_abort_grace_sec").value)
        if self.variant == "feas_only":
            self._cv_abort_grace_sec = 0.0
        self._press_deadline: Optional[float] = None

        # ── Auto-commit ──────────────────────────────────────────────
        self.declare_parameter("auto_commit", True)
        self.declare_parameter("auto_commit_dwell_sec", 1.0)
        self.declare_parameter("hold_timeout_sec", 5.0)
        self.declare_parameter("capture_settle_sec", 0.8)
        self.auto_commit       = bool(self.get_parameter("auto_commit").value)
        self.auto_commit_dwell = float(self.get_parameter("auto_commit_dwell_sec").value)
        self.hold_timeout      = float(self.get_parameter("hold_timeout_sec").value)
        self.capture_settle    = float(self.get_parameter("capture_settle_sec").value)
        self._auto_committed   = False
        self._hold_entered_mono: Optional[float] = None
        self._hold_had_targets: bool = False
        self._hold_capture_start: Optional[float] = None

        # ── Press geometry params ────────────────────────────────────
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("ee_link", "end_effector_link")
        self.declare_parameter("tool_forward_axis", "z")
        self.declare_parameter("tool_up_axis", "y")
        self.declare_parameter("approach_in", 15.0)
        self.declare_parameter("hover_in", 1.5)
        self.declare_parameter("tip_offset_in", 6.0)
        self.declare_parameter("home_named_target", "home")
        self.declare_parameter("press_mode", "snapshot_sequence")
        self.declare_parameter("servo_loop_hz", 4.0)
        self.declare_parameter("servo_align_timeout_sec", 12.0)
        self.declare_parameter("servo_press_timeout_sec", 8.0)
        self.declare_parameter("servo_step_m", 0.03)
        self.declare_parameter("servo_press_step_m", 0.012)
        self.declare_parameter("servo_align_tol_m", 0.008)
        self.declare_parameter("servo_pair_loss_grace_sec", 1.5)
        self.declare_parameter("allow_white_only_servo", True)
        self.declare_parameter("servo_lock_orientation", True)
        self.declare_parameter("servo_go_home_after_press", True)
        self.declare_parameter("try_pilz_lin", True)
        self.declare_parameter("pilz_namespace", "pilz_lin")
        self.press_mode = str(self.get_parameter("press_mode").value).strip()
        self.servo_loop_hz = max(1.0, float(self.get_parameter("servo_loop_hz").value))
        self.servo_align_timeout_sec = float(self.get_parameter("servo_align_timeout_sec").value)
        self.servo_press_timeout_sec = float(self.get_parameter("servo_press_timeout_sec").value)
        self.servo_step_m = max(0.001, float(self.get_parameter("servo_step_m").value))
        self.servo_press_step_m = max(0.001, float(self.get_parameter("servo_press_step_m").value))
        self.servo_align_tol_m = max(0.001, float(self.get_parameter("servo_align_tol_m").value))
        self.servo_pair_loss_grace_sec = max(0.0, float(self.get_parameter("servo_pair_loss_grace_sec").value))
        self.allow_white_only_servo = bool(self.get_parameter("allow_white_only_servo").value)
        if self.variant == "feas_only":
            self.allow_white_only_servo = False
            self.servo_press_step_m = min(self.servo_press_step_m, 0.004)
        self.servo_lock_orientation = bool(self.get_parameter("servo_lock_orientation").value)
        self.servo_go_home_after_press = bool(self.get_parameter("servo_go_home_after_press").value)

        # ── MoveIt params ────────────────────────────────────────────
        self.declare_parameter("robot_name", "kinova_gen3_6dof_robotiq_2f_85")
        self.declare_parameter("moveit_config_pkg", "kinova_gen3_6dof_robotiq_2f_85_moveit_config")
        self.declare_parameter("move_group_name", "manipulator")

        # ── Collision object params ──────────────────────────────────
        self.declare_parameter("collision_enabled", True)
        self.declare_parameter("table_point", [-0.3556 - 0.12 - 0.04, 0.5, 0.14605 + 0.015])
        self.declare_parameter("table_yaw", math.pi / 2.0)
        self.declare_parameter("wheelchair_point", [0.0, -0.39, 0.12065])
        self.declare_parameter("wheelchair_yaw", 0.0)

        # ── Perception state ─────────────────────────────────────────
        self._perception_lock = threading.Lock()
        self.last_quality: Optional[list] = None
        self.fan_ok_current    = False
        self.fan_ok_true_since = None
        self.f_cv_current      = False
        self.f_cv_true_since   = None
        self._f_cv_false_since: Optional[float] = time.monotonic()

        self._white_xyz: Optional[np.ndarray] = None
        self._gray_xyz:  Optional[np.ndarray] = None
        self._white_stamp: Optional[float] = None
        self._gray_stamp:  Optional[float] = None
        self._paired_white_xyz: Optional[np.ndarray] = None
        self._paired_gray_xyz: Optional[np.ndarray] = None
        self._paired_stamp: Optional[float] = None
        self._last_valid_fwd: Optional[np.ndarray] = None

        # ── Depth-based obstacle gate state ──────────────────────────
        self._obstacle_lock = threading.Lock()
        self._obstacle_blocked = False
        self._obstacle_closest_dist = float("inf")
        self._obstacle_closest_hist = deque(maxlen=self.obstacle_median_window)
        self._obstacle_stop_hits = 0
        self._obstacle_clear_hits = 0
        self._obstacle_last_depth_mono = time.monotonic()
        self._obstacle_last_clear_mono = time.monotonic()
        self._obstacle_timeout_active = False
        self._obstacle_bridge = CvBridge() if self._obstacle_gate_enabled else None

        # NEW: Dynamic Phase-Aware bandpass distance
        self._obstacle_mode = "travel"
        self._obstacle_ignore_beyond = 5.0
        
        self._scene_collision_lock = threading.Lock()
        self._table_collision_obj: Optional[CollisionObjectMsg] = None
        self._table_collision_present = False
        self._table_collision_z_shift = 0.0 
        
        self._execute_attempt_count = 0
        self._object_avoidance_enabled = True

        # ── State machine ────────────────────────────────────────────
        self.state = PRE_SCAN if self.enable_prescan else HOLD
        if self.state == HOLD:
            self._hold_entered_mono = time.monotonic()
        self.scan_idx = 0
        self.scan_reached_mono = None
        self.scan_retry_after_mono = 0.0
        self.prescan_cycles = 0
        self._fan_found_scan_idx: Optional[int] = None

        # ── Press thread ─────────────────────────────────────────────
        self._press_thread:  Optional[threading.Thread] = None
        self._press_result:  Optional[Tuple[bool, str]]  = None
        self._cv_aborted    = False
        self._obstacle_aborted = False
        self._t_start_mono: Optional[float] = None
        self._commit_only_attempted = False
        self._vision_resumed_once = False

        # ── Subscriptions ────────────────────────────────────────────
        self.create_subscription(Float32MultiArray, "/fan/quality", self._on_quality, 10, callback_group=self._sub_cg)
        self.create_subscription(Float32MultiArray, "/fan/buttons", self._on_buttons, 10, callback_group=self._sub_cg)
        self.create_subscription(WrenchStamped, self.force_topic, self._on_wrench, 10, callback_group=self._sub_cg)
        
        if self._obstacle_gate_enabled:
            self.create_subscription(Image, self.obstacle_depth_topic, self._on_obstacle_depth, 10, callback_group=self._sub_cg)
            self.create_subscription(CameraInfo, self.obstacle_info_topic, self._on_obstacle_info, 10, callback_group=self._sub_cg)
            self.get_logger().info(
                f"Obstacle gate enabled (stop={self.obstacle_stop_distance_m:.3f}m, "
                f"resume={self.obstacle_resume_distance_m:.3f}m, depth_topic={self.obstacle_depth_topic})")

        # ── Service server ───────────────────────────────────────────
        self.create_service(Trigger, "/commit_press", self._on_commit, callback_group=self._srv_cg)

        # ── Metric publishers ────────────────────────────────────────
        self.pub_state = self.create_publisher(Float32MultiArray, "/fan/sup_state", 10)
        self.pub_event = self.create_publisher(Float32MultiArray, "/fan/sup_event", 10)

        # ── MoveItPy ─────────────────────────────────────────────────
        rn  = str(self.get_parameter("robot_name").value)
        pkg = str(self.get_parameter("moveit_config_pkg").value)
        grp = str(self.get_parameter("move_group_name").value)
        self._group_name = grp

        self._moveit = None
        self._pc     = None
        self._pilz_params = None
        self._default_planning_pipeline = "ompl"
        try:
            mcfg = MoveItConfigsBuilder(rn, package_name=pkg).to_moveit_configs()
            cd = _augment_cfg(mcfg.to_dict())
            self._default_planning_pipeline = str(cd.get("default_planning_pipeline", "ompl"))
            self._moveit = MoveItPy(node_name="fan_sup_moveitpy", config_dict=cd)
            self._pc = self._moveit.get_planning_component(grp)
            self.get_logger().info(f"MoveItPy ready ({rn} / {grp})")
            if bool(self.get_parameter("try_pilz_lin").value):
                ns = self.get_parameter("pilz_namespace").value
                try:
                    self._pilz_params = PlanRequestParameters(self._moveit, ns)
                    self._pilz_params.planning_pipeline = "pilz_industrial_motion_planner"
                    self._pilz_params.planner_id = "LIN"
                    self._pilz_params.planning_time = 2.0
                    self._pilz_params.planning_attempts = 1
                    self._pilz_params.max_velocity_scaling_factor = 0.05
                    self._pilz_params.max_acceleration_scaling_factor = 0.05
                    self.get_logger().info(f"Pilz LIN enabled (ns={ns})")
                except Exception as e:
                    self.get_logger().warn(f"Pilz init failed: {e}")
        except Exception as e:
            self.get_logger().error(f"MoveItPy init failed: {e}")

        if self._moveit is not None and bool(self.get_parameter("collision_enabled").value):
            self._setup_collision_objects()

        # ── Timers ───────────────────────────────────────────────────
        self.create_timer(0.05, self._tick)
        self.create_timer(0.05, self._publish_state_msg, callback_group=self._sub_cg)

        self.get_logger().info(
            f"FanPressSupervisor up — variant={self.variant} state={self.state} "
            f"single_shot={self.commit_only_single_shot} trial={self.trial_id}")

    def _trial_t(self) -> float:
        return time.monotonic() - self.trial_t0

    # =================================================================
    #  Collision Helpers
    # =================================================================
    @staticmethod
    def _quat_multiply(a, b):
        ax, ay, az, aw = a; bx, by, bz, bw = b
        x = aw * bx + ax * bw + ay * bz - az * by
        y = aw * by - ax * bz + ay * bw + az * bx
        z = aw * bz + ax * by - ay * bx + az * bw
        w = aw * bw - ax * bx - ay * by - az * bz
        return (x, y, z, w)

    @staticmethod
    def _rotate_vec_by_quat(v, q):
        vx, vy, vz = v; qx, qy, qz, qw = q
        ix = qw * vx + qy * vz - qz * vy; iy = qw * vy + qz * vx - qx * vz
        iz = qw * vz + qx * vy - qy * vx; iw = -qx * vx - qy * vy - qz * vz
        rx = ix * qw + iw * -qx + iy * -qz - iz * -qy
        ry = iy * qw + iw * -qy + iz * -qx - ix * -qz
        rz = iz * qw + iw * -qz + ix * -qy - iy * -qx
        return (rx, ry, rz)

    def _apply_base_to_local(self, base_pose: Pose, local_pose: Pose) -> Pose:
        out = Pose()
        bq = (base_pose.orientation.x, base_pose.orientation.y, base_pose.orientation.z, base_pose.orientation.w)
        local_pos = (local_pose.position.x, local_pose.position.y, local_pose.position.z)
        rotated = self._rotate_vec_by_quat(local_pos, bq)
        out.position.x = base_pose.position.x + rotated[0]
        out.position.y = base_pose.position.y + rotated[1]
        out.position.z = base_pose.position.z + rotated[2]
        lq = (local_pose.orientation.x, local_pose.orientation.y, local_pose.orientation.z, local_pose.orientation.w)
        rx, ry, rz, rw = self._quat_multiply(bq, lq)
        out.orientation.x = rx; out.orientation.y = ry; out.orientation.z = rz; out.orientation.w = rw
        return out

    @staticmethod
    def _make_box(size: List[float]) -> SolidPrimitive:
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX; box.dimensions = list(size)
        return box

    @staticmethod
    def _make_collision_pose(position, yaw: float = 0.0) -> Pose:
        p = Pose()
        p.position.x, p.position.y, p.position.z = position
        half = yaw / 2.0
        p.orientation.x = 0.0; p.orientation.y = 0.0; p.orientation.z = math.sin(half); p.orientation.w = math.cos(half)
        return p

    def _setup_collision_objects(self):
        try:
            table_point = list(self.get_parameter("table_point").value)
            table_yaw = float(self.get_parameter("table_yaw").value)
            wheelchair_point = list(self.get_parameter("wheelchair_point").value)
            wheelchair_yaw = float(self.get_parameter("wheelchair_yaw").value)

            base_table_pose = self._make_collision_pose(table_point, table_yaw)
            base_wheel_pose = self._make_collision_pose(wheelchair_point, wheelchair_yaw)

            # Table + Shelf 
            table_size = [0.762 + 0.0127, 1.8288, 0.2921 + 0.0127]
            table_abs_pose = self._make_collision_pose((0.6985, 0.3556, 0.14605))
            shelf_outer_size = [0.3048, 1.1176, 0.5842]
            table_edge_to_shelf = 0.4572 - 0.0127
            shelf_origin_x = 0.3175 + table_edge_to_shelf
            shelf_origin_y = 0.0 - 0.025
            shelf_origin_z = table_size[2] + 0.0127
            shelf_thickness = 0.02
            half_shelf_thickness = shelf_thickness / 2.0
            bottom_to_shelf1 = 0.073025
            bottom_to_shelf2 = 0.23495
            bottom_to_top = 0.5842
            shelf_heights = [bottom_to_shelf1 - half_shelf_thickness, bottom_to_shelf2 - half_shelf_thickness, bottom_to_top - half_shelf_thickness]
            shelf1_to_shelf2 = bottom_to_shelf2 - bottom_to_shelf1
            shelf2_to_top = bottom_to_top - bottom_to_shelf2
            left_to_part1 = 0.37465

            table_ref = (table_abs_pose.position.x, table_abs_pose.position.y, table_abs_pose.position.z)
            table_prims: List[Tuple[SolidPrimitive, Pose]] = []
            table_prims.append((self._make_box(table_size), table_abs_pose))

            for z_rel in shelf_heights:
                z = shelf_origin_z + z_rel + shelf_thickness / 2
                table_prims.append((self._make_box([shelf_outer_size[0], shelf_outer_size[1], shelf_thickness]),
                                    self._make_collision_pose((shelf_origin_x + shelf_outer_size[0] / 2, 0.0, z))))

            table_prims.append((self._make_box([shelf_outer_size[0], 0.02, shelf_outer_size[2]]),
                                self._make_collision_pose((shelf_origin_x + shelf_outer_size[0] / 2, shelf_origin_y - shelf_outer_size[1] / 2, shelf_origin_z + shelf_outer_size[2] / 2))))
            table_prims.append((self._make_box([shelf_outer_size[0], 0.02, shelf_outer_size[2]]),
                                self._make_collision_pose((shelf_origin_x + shelf_outer_size[0] / 2, shelf_origin_y + shelf_outer_size[1] / 2, shelf_origin_z + shelf_outer_size[2] / 2))))
            table_prims.append((self._make_box([0.02, shelf_outer_size[1], shelf_outer_size[2]]),
                                self._make_collision_pose((shelf_origin_x + shelf_outer_size[0] - 0.01, 0.0, shelf_origin_z + shelf_outer_size[2] / 2))))
            table_prims.append((self._make_box([0.02, shelf_outer_size[1], bottom_to_shelf1 - half_shelf_thickness]),
                                self._make_collision_pose((shelf_origin_x, 0.0, shelf_origin_z + bottom_to_shelf1 / 2))))

            middle_to_part = (shelf_outer_size[1] / 2) - left_to_part1
            table_prims.append((self._make_box([shelf_outer_size[0], 0.02, shelf1_to_shelf2]),
                                self._make_collision_pose((shelf_origin_x + shelf_outer_size[0] / 2, middle_to_part, shelf_origin_z + shelf1_to_shelf2))))
            table_prims.append((self._make_box([shelf_outer_size[0], 0.02, shelf1_to_shelf2]),
                                self._make_collision_pose((shelf_origin_x + shelf_outer_size[0] / 2, -middle_to_part, shelf_origin_z + shelf1_to_shelf2))))
            table_prims.append((self._make_box([shelf_outer_size[0], 0.02, shelf2_to_top]),
                                self._make_collision_pose((shelf_origin_x + shelf_outer_size[0] / 2, 0.0, shelf_origin_z + bottom_to_shelf1 + shelf2_to_top - shelf_thickness / 2))))

            co_table = CollisionObjectMsg()
            co_table.id = "table_shelf"
            co_table.header.frame_id = "base_link"
            co_table.operation = CollisionObjectMsg.ADD

            for prim, abs_pose in table_prims:
                local = Pose()
                local.position.x = abs_pose.position.x - table_ref[0]
                local.position.y = abs_pose.position.y - table_ref[1]
                local.position.z = abs_pose.position.z - table_ref[2]
                local.orientation = abs_pose.orientation
                world_pose = self._apply_base_to_local(base_table_pose, local)
                co_table.primitives.append(prim)
                co_table.primitive_poses.append(world_pose)

            # Wheelchair
            wall_size = [0.51, 0.5, 0.2413]
            wall_y = -0.39
            wall_z = wall_size[2] / 2.0
            wall_abs_pose = self._make_collision_pose((0.0, wall_y, wall_z))
            wheel_ref = (wall_abs_pose.position.x, wall_abs_pose.position.y, wall_abs_pose.position.z)

            co_wheel = CollisionObjectMsg()
            co_wheel.id = "wheelchair"
            co_wheel.header.frame_id = "base_link"
            co_wheel.operation = CollisionObjectMsg.ADD

            for prim, abs_pose in [(self._make_box(wall_size), wall_abs_pose)]:
                local = Pose()
                local.position.x = abs_pose.position.x - wheel_ref[0]
                local.position.y = abs_pose.position.y - wheel_ref[1]
                local.position.z = abs_pose.position.z - wheel_ref[2]
                local.orientation = abs_pose.orientation
                world_pose = self._apply_base_to_local(base_wheel_pose, local)
                co_wheel.primitives.append(prim)
                co_wheel.primitive_poses.append(world_pose)

            with self._moveit.get_planning_scene_monitor().read_write() as scene:
                scene.apply_collision_object(co_table)
                scene.apply_collision_object(co_wheel)
            with self._scene_collision_lock:
                self._table_collision_obj = deepcopy(co_table)
                self._table_collision_obj.operation = CollisionObjectMsg.ADD
                self._table_collision_present = True
                self._table_collision_z_shift = 0.0

            self.get_logger().info("Collision objects mapped.")
        except Exception as e:
            self.get_logger().error(f"Failed to setup collision objects: {e}")

    def _update_object_avoidance_policy(self):
        enabled = (self._execute_attempt_count <= 1)
        if enabled == self._object_avoidance_enabled: return
        self._object_avoidance_enabled = enabled
        if enabled:
            self.get_logger().info("Object avoidance ENABLED (execute attempt #1).")
            self._setup_collision_objects()
        else:
            self.get_logger().warn("Object avoidance SHRINKING table boundaries (execute attempt #2+).")
            # Don't delete everything, just drop the table 4cm to allow surface-level planning
            self._shift_table_collision_z(-0.04, "execute attempt #2+ policy")

    def _shift_table_collision_z(self, z_offset_m: float, context: str = "") -> bool:
        if self._moveit is None or not bool(self.get_parameter("collision_enabled").value):
            return True
        with self._scene_collision_lock:
            if self._table_collision_obj is None:
                return False
            try:
                obj = deepcopy(self._table_collision_obj)
                obj.operation = CollisionObjectMsg.ADD
                for pose in obj.primitive_poses:
                    pose.position.z += z_offset_m
                with self._moveit.get_planning_scene_monitor().read_write() as scene:
                    scene.apply_collision_object(obj)
                self._table_collision_z_shift = z_offset_m
                self.get_logger().info(f"Table collision Z shifted by {z_offset_m:.3f}m (phase: {context})")
                return True
            except Exception as e:
                self.get_logger().error(f"Failed to shift table collision: {e}")
                return False

    def _restore_table_collision(self, context: str = ""):
        if not self._object_avoidance_enabled: return True
        return self._shift_table_collision_z(0.0, context)

    # =================================================================
    #  Callbacks 
    # =================================================================
    
    def _on_wrench(self, msg: WrenchStamped):
        """Monitors end-effector forces to prevent Kortex hardware faults."""
        # Simple Euclidean limit, or isolate Z if pushing along tool frame
        f_norm = math.sqrt(msg.wrench.force.x**2 + msg.wrench.force.y**2 + msg.wrench.force.z**2)
        if f_norm > self.max_force_n:
            with self._force_lock:
                if not self._force_exceeded:
                    self.get_logger().error(f"FORCE LIMIT EXCEEDED: {f_norm:.1f}N > {self.max_force_n}N")
                self._force_exceeded = True

    def _on_quality(self, msg: Float32MultiArray):
        with self._perception_lock:
            self.last_quality = list(msg.data)
            self._update_fan_ok()
            self._update_feasibility()

    def _on_buttons(self, msg: Float32MultiArray):
        if len(msg.data) < 6: return
        wx, wy, wz, gx, gy, gz = [float(v) for v in msg.data[:6]]
        white_ok = all(math.isfinite(v) for v in (wx, wy, wz))
        gray_ok = all(math.isfinite(v) for v in (gx, gy, gz))
        now_mono = time.monotonic()
        pair_dist = None
        pair_valid = False
        if white_ok and gray_ok:
            wv = np.array([wx, wy, wz], dtype=float)
            gv = np.array([gx, gy, gz], dtype=float)
            pair_dist = float(np.linalg.norm(wv - gv))
            pair_valid = (self.button_pair_min_dist_m <= pair_dist <= self.button_pair_max_dist_m)
        with self._perception_lock:
            if white_ok:
                self._white_xyz = np.array([wx, wy, wz], dtype=float); self._white_stamp = now_mono
            elif not self.retain_last_buttons_on_nan:
                self._white_xyz = None; self._white_stamp = None
            if gray_ok:
                self._gray_xyz = np.array([gx, gy, gz], dtype=float); self._gray_stamp = now_mono
            elif not self.retain_last_buttons_on_nan:
                self._gray_xyz = None; self._gray_stamp = None
            if white_ok and gray_ok and pair_valid:
                self._paired_white_xyz = np.array([wx, wy, wz], dtype=float)
                self._paired_gray_xyz = np.array([gx, gy, gz], dtype=float)
                self._paired_stamp = now_mono
                try: self._last_valid_fwd = _compute_forward(self._paired_white_xyz, self._paired_gray_xyz)
                except Exception: pass

    def _on_obstacle_info(self, _msg: CameraInfo):
        pass

    def _on_obstacle_depth(self, msg: Image):
        """Process depth for obstacle detection using a Bandpass approach."""
        if not self._obstacle_gate_enabled or self._obstacle_bridge is None: return
        try: depth_raw = self._obstacle_bridge.imgmsg_to_cv2(msg, "passthrough")
        except Exception: return

        d = depth_raw.astype(np.float32)
        if msg.encoding.lower() in ("16uc1", "mono16"): d /= 1000.0
        if d.ndim < 2: return

        h, w = d.shape[:2]
        rh = int(h * self.obstacle_roi_fraction / 2.0)
        rw = int(w * self.obstacle_roi_fraction / 2.0)
        cy, cx = h // 2, w // 2
        if rh < 1 or rw < 1: return
        roi = d[cy - rh:cy + rh, cx - rw:cx + rw]
        roi_total_px = max(1, int(roi.size))

        # Bandpass Filter: ignore depths > _obstacle_ignore_beyond so we don't trigger on the fan itself
        valid = roi[np.isfinite(roi) & (roi > 0.01) & (roi < self._obstacle_ignore_beyond)]
        valid_ratio = float(valid.size) / float(roi_total_px)
        now = time.monotonic()

        with self._obstacle_lock:
            self._obstacle_last_depth_mono = now

            if (valid.size >= self.obstacle_min_valid_pixels and valid_ratio >= self.obstacle_min_valid_ratio):
                raw_closest = float(np.percentile(valid, self.obstacle_percentile))
                self._obstacle_closest_hist.append(raw_closest)
                closest = float(np.median(np.asarray(self._obstacle_closest_hist, dtype=float)))
                self._obstacle_closest_dist = closest
                was_blocked = self._obstacle_blocked

                if closest < self.obstacle_stop_distance_m:
                    self._obstacle_stop_hits += 1; self._obstacle_clear_hits = 0
                elif closest > self.obstacle_resume_distance_m:
                    self._obstacle_clear_hits += 1; self._obstacle_stop_hits = 0
                else:
                    self._obstacle_stop_hits = 0; self._obstacle_clear_hits = 0

                if not was_blocked and self._obstacle_stop_hits >= self.obstacle_stop_confirm_frames:
                    self._obstacle_blocked = True; self._obstacle_stop_hits = 0
                    self.get_logger().warn(f"OBSTACLE gate: closest={closest:.3f}m < stop={self.obstacle_stop_distance_m:.3f}m")
                elif was_blocked and self._obstacle_clear_hits >= self.obstacle_resume_confirm_frames:
                    self._obstacle_blocked = False; self._obstacle_clear_hits = 0; self._obstacle_last_clear_mono = now
            else:
                self._obstacle_closest_dist = float("inf")
                # Failsafe triggers only in Travel Mode
                if self.obstacle_fail_safe_on_no_depth and self._obstacle_mode == "travel":
                    self._obstacle_stop_hits += 1; self._obstacle_clear_hits = 0
                    if not self._obstacle_blocked and self._obstacle_stop_hits >= self.obstacle_stop_confirm_frames:
                        self._obstacle_blocked = True; self._obstacle_stop_hits = 0
                else:
                    self._obstacle_clear_hits += 1; self._obstacle_stop_hits = 0
                    if self._obstacle_blocked and self._obstacle_clear_hits >= self.obstacle_resume_confirm_frames:
                        self._obstacle_blocked = False; self._obstacle_clear_hits = 0; self._obstacle_last_clear_mono = now

    def _obstacle_is_blocked(self) -> bool:
        if not self._object_avoidance_enabled or not self._obstacle_gate_enabled: return False
        now = time.monotonic()
        with self._obstacle_lock:
            blocked = self._obstacle_blocked
            if self.obstacle_fail_safe_on_no_depth and self.obstacle_depth_timeout_sec > 0.0 and self._obstacle_mode == "travel":
                age = now - self._obstacle_last_depth_mono
                if age > self.obstacle_depth_timeout_sec:
                    blocked = True
                    if not self._obstacle_timeout_active:
                        self._obstacle_timeout_active = True
                        self._obstacle_blocked = True
                        self._obstacle_closest_dist = float("inf")
                        self._obstacle_stop_hits = 0; self._obstacle_clear_hits = 0
                elif self._obstacle_timeout_active:
                    self._obstacle_timeout_active = False; self._obstacle_last_clear_mono = now
            return blocked

    def _set_obstacle_phase(self, mode: str, context: str = ""):
        """Sets bandpass filter thresholds based on operation phase."""
        self._obstacle_mode = mode
        if mode == "travel":
            self._obstacle_ignore_beyond = 5.0
        else: # "press"
            self._obstacle_ignore_beyond = self.obstacle_stop_distance_m - 0.02
        self.get_logger().info(f"Obstacle gate set to {mode} (max_depth={self._obstacle_ignore_beyond:.3f}m) phase: {context}")

    def _wait_for_obstacle_clear(self, context: str) -> bool:
        if not self._object_avoidance_enabled or not self._obstacle_gate_enabled: return True
        if self._execute_attempt_count == 1 and self.state == EXECUTE:
            return False # Force replan on first attempt
            
        t0 = time.monotonic()
        self.get_logger().warn(f"PAUSE: obstacle gate at '{context}' — waiting...")
        while rclpy.ok():
            if not self._obstacle_is_blocked():
                if self.obstacle_clear_hold_sec > 0.0:
                    with self._obstacle_lock:
                        since_clear = time.monotonic() - self._obstacle_last_clear_mono
                    if since_clear < self.obstacle_clear_hold_sec:
                        time.sleep(min(0.05, self.obstacle_clear_hold_sec - since_clear)); continue
                return True
            if self._press_deadline is not None and time.monotonic() > self._press_deadline: return False
            if self.obstacle_pause_timeout_sec > 0.0 and (time.monotonic() - t0) > self.obstacle_pause_timeout_sec: return False
            time.sleep(0.05)
        return False

    def _maybe_wait_for_obstacle_gate(self, context: str) -> bool:
        if not self._object_avoidance_enabled or self.state != EXECUTE: return True
        if not self._obstacle_is_blocked(): return True
        return self._wait_for_obstacle_clear(context)

    # =================================================================
    #  Perception State Updates
    # =================================================================
    def _update_fan_ok(self):
        q = self.last_quality
        if not q or len(q) < Q_MIN_LEN: self._set_fan_ok(False); return
        self._set_fan_ok(q[Q_FAN_CONF] >= self.min_fan_conf)

    def _set_fan_ok(self, v: bool):
        now = self.get_clock().now()
        if v and not self.fan_ok_current: self.fan_ok_true_since = now
        if not v and self.fan_ok_current: self.fan_ok_true_since = None
        self.fan_ok_current = v

    def _stable_fan(self, dwell: float) -> bool:
        if not self.fan_ok_current or self.fan_ok_true_since is None: return False
        return (self.get_clock().now() - self.fan_ok_true_since) >= Duration(seconds=dwell)

    def _update_feasibility(self):
        q = self.last_quality
        if not q or len(q) < Q_MIN_LEN: self._set_f_cv(False); return
        ok = (q[Q_TF_OK] > 0.5 and q[Q_W_CONF] >= self.min_btn_conf and q[Q_G_CONF] >= self.min_btn_conf
              and q[Q_W_VR] >= self.min_valid_ratio and q[Q_G_VR] >= self.min_valid_ratio
              and q[Q_W_USED] >= self.min_used_px and q[Q_G_USED] >= self.min_used_px)
        self._set_f_cv(ok)

    def _set_f_cv(self, v: bool):
        now = self.get_clock().now()
        now_mono = time.monotonic()
        if v and not self.f_cv_current:
            self.f_cv_true_since = now; self._f_cv_false_since = None
        if not v and self.f_cv_current:
            self.f_cv_true_since = None; self._f_cv_false_since = now_mono
        if not v and self._f_cv_false_since is None:
            self._f_cv_false_since = now_mono
        self.f_cv_current = v

    def _stable_feasible(self) -> bool:
        if not self.f_cv_current or self.f_cv_true_since is None: return False
        return (self.get_clock().now() - self.f_cv_true_since) >= Duration(seconds=self.dwell_sec)

    def _buttons_available(self) -> bool:
        now = time.monotonic()
        with self._perception_lock:
            if self._paired_white_xyz is None or self._paired_gray_xyz is None or self._paired_stamp is None: return False
            age_pair = now - self._paired_stamp
        return age_pair < self._btn_stale_sec

    def _white_available(self) -> bool:
        now = time.monotonic()
        with self._perception_lock:
            if self._white_xyz is None or self._white_stamp is None: return False
            age_white = now - self._white_stamp
        return age_white < self._btn_stale_sec

    def _execute_target_available(self) -> bool:
        if self._buttons_available(): return True
        return (self.press_mode == "visual_servo_white" and self.allow_white_only_servo and self._white_available())

    def _pair_visible_recent(self, max_age_sec: float) -> bool:
        now = time.monotonic()
        with self._perception_lock:
            if self._paired_stamp is None: return False
            age_pair = now - self._paired_stamp
        return age_pair <= max_age_sec

    def _publish_state_msg(self):
        q = self.last_quality
        fc = float(q[Q_FAN_CONF]) if (q and len(q) >= Q_MIN_LEN) else 0.0
        m = Float32MultiArray()
        m.data = [float(self.trial_id), float(VARIANT_ID[self.variant]), float(STATE_ID.get(self.state, -1)), 1.0 if self.f_cv_current else 0.0, fc]
        self.pub_state.publish(m)

    def _log_event(self, started: int, t_start: float, t_success: float, outcome: int):
        m = Float32MultiArray()
        m.data = [float(self.trial_id), float(VARIANT_ID[self.variant]), float(started), t_start, t_success, float(outcome)]
        self.pub_event.publish(m)

    # =================================================================
    #  Actions & Sequencing
    # =================================================================
    def _on_commit(self, req, res):
        if self.state != HOLD:
            res.success = False; res.message = f"Cannot commit in state {self.state}"; return res
        if self.variant == "feas_only":
            res.success = False; res.message = "feas_only auto-starts"; return res
        if self._hold_capture_start is not None:
            elapsed = time.monotonic() - self._hold_capture_start
            remaining = self.capture_settle - elapsed
            if remaining > 0:
                res.success = False; res.message = "Fresh-capture window active"; return res
        res.success = True; res.message = self._do_commit()
        return res

    def _do_commit(self) -> str:
        if self.variant == "commit_only":
            if self.commit_only_single_shot and self._commit_only_attempted: return "Attempted."
            if not self._buttons_available(): return "Blocked."
            self._start_execute(); return "Started."
        if self.variant == "hold_commit":
            if self.f_cv_current and self._execute_target_available():
                self._start_execute(); return "Started."
            self._log_event(0, self._trial_t(), float("nan"), OUTCOME_ABORT); self._auto_committed = False; return "Blocked."
        if self.variant == "hac":
            if self._execute_target_available() and (self._stable_feasible() or self.hac_allow_latched_execute):
                self._start_execute(); return "Started."
            start_idx = self._fan_found_scan_idx if self._fan_found_scan_idx is not None else 0
            self._enter_assist(start_scan_idx=start_idx); return "Entered ASSIST."
        return "Unknown"

    def _hold_track_occlusion(self):
        targets_now = self._execute_target_available()
        if self._hold_had_targets and not targets_now:
            with self._perception_lock:
                self._paired_white_xyz = None; self._paired_gray_xyz = None; self._paired_stamp = None
            self._auto_committed = False; self.f_cv_true_since = None; self._hold_capture_start = None
        if not self._hold_had_targets and targets_now:
            self._hold_capture_start = time.monotonic(); self.f_cv_true_since = None
        self._hold_had_targets = targets_now
        if self._hold_capture_start is not None:
            if (time.monotonic() - self._hold_capture_start) < self.capture_settle: return True
            self._hold_capture_start = None
        return False

    def _enter_hold(self):
        self.state = HOLD; self._hold_entered_mono = time.monotonic()
        self._hold_had_targets = self._execute_target_available()
        self._hold_capture_start = None
        self._set_obstacle_phase("travel", "enter_hold")
        with self._force_lock: self._force_exceeded = False

    def _enter_assist(self, start_scan_idx: int = 0):
        self.state = ASSIST; self.scan_idx = start_scan_idx
        self.scan_reached_mono = None; self.scan_retry_after_mono = 0.0
        self.assist_cycles = 0; self.f_cv_current = False; self.f_cv_true_since = None
        self._set_obstacle_phase("travel", "enter_assist")
        with self._force_lock: self._force_exceeded = False

    def _start_execute(self, is_retry: bool = False):
        self.state = EXECUTE
        if not is_retry: self._execute_attempt_count = 0
        self._execute_attempt_count += 1
        self._update_object_avoidance_policy()
        self._t_start_mono = time.monotonic()
        self._press_result = None; self._cv_aborted = False; self._obstacle_aborted = False; self._vision_resumed_once = False
        with self._force_lock: self._force_exceeded = False
        self._set_obstacle_phase("travel", "start_execute")
        if self.variant == "commit_only": self._commit_only_attempted = True
        self._press_deadline = time.monotonic() + self._press_timeout_sec
        self._press_thread = threading.Thread(target=self._run_press_sequence, daemon=True)
        self._press_thread.start()

    def _wait_for_vision_resume(self, context: str) -> bool:
        t0 = time.monotonic()
        while rclpy.ok():
            if self._execute_target_available():
                self._vision_resumed_once = True; return True
            if self._press_deadline is not None and time.monotonic() > self._press_deadline: return False
            if self.pause_vision_timeout_sec > 0.0 and (time.monotonic() - t0) > self.pause_vision_timeout_sec: return False
            time.sleep(0.05)
        return False

    def _execute_trajectory(self, trajectory) -> bool:
        res = self._moveit.execute(trajectory, controllers=[])
        if res is None: return True
        if isinstance(res, bool): return res
        success = getattr(res, "success", None)
        if isinstance(success, bool): return success
        error_code = getattr(res, "error_code", None)
        if error_code is not None:
            val = getattr(error_code, "val", None)
            if val is not None: return int(val) == MoveItErrorCodes.SUCCESS
        return True

    def _move_to_scan_pose(self, phase: str) -> bool:
        now = time.monotonic()
        if now < self.scan_retry_after_mono: return False
        p = list(SCAN_POSES[self.scan_idx])
        if len(p) >= 2 and self.scan_joint2_limit_rad > 0.0:
            j2_raw = float(p[1])
            p[1] = max(-self.scan_joint2_limit_rad, min(self.scan_joint2_limit_rad, j2_raw))
        ok = self._plan_exec_joints(p)
        if ok: self.scan_reached_mono = now; return True
        self.scan_reached_mono = None; self.scan_retry_after_mono = now + 1.0; return False

    def _plan_exec_joints(self, joints: List[float], pt: float = 2.8) -> bool:
        if self._moveit is None: return False
        if not self._maybe_wait_for_obstacle_gate("plan_exec_joints"): return False
        try:
            self._pc.set_start_state_to_current_state()
            gs = RobotState(self._moveit.get_robot_model())
            gs.set_joint_group_positions(self._group_name, joints)
            gs.update()
            self._pc.set_goal_state(robot_state=gs)
            params = PlanRequestParameters(self._moveit, "")
            params.planning_pipeline = self._default_planning_pipeline
            params.planning_time = pt; params.planning_attempts = 1
            r = self._pc.plan(single_plan_parameters=params)
            if not r: return False
            return self._execute_trajectory(r.trajectory)
        except Exception: return False

    def _trajectory_joint2_limits_ok(self, trajectory, context: str) -> bool:
        limit = self.scan_joint2_limit_rad
        if limit <= 0.0: return True
        jt = getattr(trajectory, "joint_trajectory", None)
        if jt is None: return True
        joint_names = list(getattr(jt, "joint_names", []))
        j2_idx = 1 if not joint_names else (joint_names.index("joint_2") if "joint_2" in joint_names else None)
        if j2_idx is None: return True
        for pt in getattr(jt, "points", []):
            positions = getattr(pt, "positions", [])
            if j2_idx >= len(positions): continue
            j2 = float(positions[j2_idx])
            if math.isfinite(j2) and abs(j2) > (limit + 1e-6): return False
        return True

    def _plan_exec_pose(self, pose: PoseStamped, linear: bool, enforce_joint2_limit: bool = False, context: str = "pose") -> bool:
        if self._moveit is None: return False
        if not self._maybe_wait_for_obstacle_gate("plan_exec_pose"): return False
        try:
            ee = self.get_parameter("ee_link").value
            self._pc.set_start_state_to_current_state()
            self._pc.set_goal_state(pose_stamped_msg=pose, pose_link=ee)
            if linear and self._pilz_params is not None:
                r = self._pc.plan(single_plan_parameters=self._pilz_params)
                if r: return self._execute_trajectory(r.trajectory)
            params = PlanRequestParameters(self._moveit, ""); params.planning_pipeline = self._default_planning_pipeline
            params.planning_time = 2.0; params.planning_attempts = 1
            attempts = 3 if enforce_joint2_limit else 1
            for _ in range(attempts):
                r = self._pc.plan(single_plan_parameters=params)
                if not r: continue
                if enforce_joint2_limit and not self._trajectory_joint2_limits_ok(r.trajectory, context): continue
                return self._execute_trajectory(r.trajectory)
            return False
        except Exception: return False

    def _plan_exec_named(self, name: str) -> bool:
        if self._moveit is None: return False
        if not self._maybe_wait_for_obstacle_gate(f"plan_exec_named:{name}"): return False
        try:
            self._pc.set_start_state_to_current_state()
            self._pc.set_goal_state(configuration_name=name)
            params = PlanRequestParameters(self._moveit, ""); params.planning_pipeline = self._default_planning_pipeline
            params.planning_time = 2.0; params.planning_attempts = 1
            r = self._pc.plan(single_plan_parameters=params)
            return self._execute_trajectory(r.trajectory) if r else False
        except Exception: return False

    def _make_pose(self, pos: np.ndarray, quat: Tuple[float, float, float, float]) -> PoseStamped:
        m = PoseStamped()
        m.header.frame_id = self.get_parameter("base_frame").value
        m.header.stamp = self.get_clock().now().to_msg()
        m.pose.position.x = float(pos[0]); m.pose.position.y = float(pos[1]); m.pose.position.z = float(pos[2])
        m.pose.orientation.x = quat[0]; m.pose.orientation.y = quat[1]; m.pose.orientation.z = quat[2]; m.pose.orientation.w = quat[3]
        return m

    @staticmethod
    def _step_toward(cur: np.ndarray, goal: np.ndarray, max_step: float) -> np.ndarray:
        d = goal - cur
        n = float(np.linalg.norm(d))
        return goal.copy() if (n <= max_step or n < 1e-9) else cur + d * (max_step / n)

    def _get_live_servo_geom(self) -> Optional[dict]:
        with self._perception_lock:
            pair_ok = (self._paired_white_xyz is not None and self._paired_gray_xyz is not None and self._paired_stamp is not None)
            white_ok = (self._white_xyz is not None and self._white_stamp is not None)
            p1_pair = self._paired_white_xyz.copy() if pair_ok else None
            p2_pair = self._paired_gray_xyz.copy() if pair_ok else None
            stamp_pair = float(self._paired_stamp) if pair_ok else None
            p1_white = self._white_xyz.copy() if white_ok else None
            stamp_white = float(self._white_stamp) if white_ok else None
            last_fwd = self._last_valid_fwd.copy() if self._last_valid_fwd is not None else None

        now = time.monotonic()
        mode = "pair"; p1 = None; p2 = None; age = float("nan"); pair_dist = float("nan"); fwd = None

        if pair_ok:
            age_pair = now - stamp_pair
            if age_pair <= self._btn_stale_sec:
                dist_pair = float(np.linalg.norm(p1_pair - p2_pair))
                if self.button_pair_min_dist_m <= dist_pair <= self.button_pair_max_dist_m:
                    p1 = p1_pair; p2 = p2_pair; age = age_pair; pair_dist = dist_pair
                    try: fwd = _compute_forward(p1, p2)
                    except Exception: fwd = None

        if fwd is None:
            if not (self.allow_white_only_servo and white_ok): return None
            age_white = now - stamp_white
            if age_white > self._btn_stale_sec: return None
            p1 = p1_white; p2 = p1_white.copy(); age = age_white; mode = "white_only"
            if last_fwd is not None and np.linalg.norm(last_fwd[:2]) > 1e-6:
                fwd = _normalize(np.array([last_fwd[0], last_fwd[1], 0.0], dtype=float))
            else:
                radial = np.array([p1[0], p1[1], 0.0], dtype=float)
                if np.linalg.norm(radial) < 1e-6: return None
                fwd = _normalize(radial)

        try:
            quat = _make_orientation_quat(fwd, np.array([0.0, 0.0, 1.0]), self.get_parameter("tool_forward_axis").value, self.get_parameter("tool_up_axis").value)
        except Exception: return None

        approach_m = float(self.get_parameter("approach_in").value) * INCH_TO_M
        hover_m = float(self.get_parameter("hover_in").value) * INCH_TO_M
        tip_offset_m = float(self.get_parameter("tip_offset_in").value) * INCH_TO_M
        z_hat = np.array([0.0, 0.0, 1.0])

        def ee_from_tip(tip: np.ndarray) -> np.ndarray: return tip - fwd * tip_offset_m

        return {
            "mode": mode, "p1": p1, "p2": p2, "pair_age": age, "pair_dist": pair_dist, "quat": quat,
            "ee_approach": ee_from_tip(p1 - fwd * approach_m + z_hat * hover_m),
            "ee_hover": ee_from_tip(p1 + z_hat * hover_m),
            "ee_press": ee_from_tip(p1.copy()),
        }

    def _check_abort(self, context: str, cv_phase: str = "all") -> Optional[str]:
        if self._press_deadline is not None and time.monotonic() > self._press_deadline: return f"Press timeout at '{context}'"
        
        # Priority force abort
        with self._force_lock:
            if self._force_exceeded: return "Force/Torque safety limit exceeded"

        if self._obstacle_is_blocked(): return WAIT_FOR_OBSTACLE
        
        if (self.variant == "feas_only" and cv_phase == "press"
                and not (self.resume_then_ignore_vision_loss and self._vision_resumed_once)
                and not self._pair_visible_recent(self.feas_only_live_pair_timeout_sec)):
            if self.pause_on_vision_loss: return WAIT_FOR_VISION
            return f"Buttons lost at '{context}'"
            
        if (self.variant != "commit_only" and self.abort_on_cv_loss_during_execute and not self.f_cv_current):
            if (cv_phase == "press" and self.resume_then_ignore_vision_loss and self._vision_resumed_once): return None
            if cv_phase == "press" and self.pause_on_vision_loss: return WAIT_FOR_VISION
            if self.cv_abort_phase == "press_only" and cv_phase != "press": return None
            if self._cv_abort_grace_sec > 0.0:
                if self._f_cv_false_since is None: self._f_cv_false_since = time.monotonic()
                if (time.monotonic() - self._f_cv_false_since) < self._cv_abort_grace_sec: return None
            return f"CV lost at '{context}'"
        return None

    def _run_press_visual_servo(self) -> Tuple[bool, str]:
        dt = 1.0 / max(1.0, self.servo_loop_hz)
        last_geom: Optional[dict] = None
        pair_loss_since: Optional[float] = None

        def get_geom(allow_stale: bool) -> Optional[dict]:
            nonlocal last_geom, pair_loss_since
            g = self._get_live_servo_geom()
            if g is not None:
                last_geom = g; pair_loss_since = None; return g
            if last_geom is None: return None
            if pair_loss_since is None: pair_loss_since = time.monotonic()
            if allow_stale: return last_geom
            if (time.monotonic() - pair_loss_since) > self.servo_pair_loss_grace_sec: return None
            return last_geom

        t0 = time.monotonic()
        while True:
            reason = self._check_abort("before VS init", cv_phase="align")
            if reason == WAIT_FOR_OBSTACLE:
                if not self._wait_for_obstacle_clear("before VS init"):
                    self._obstacle_aborted = True; return False, "Obstacle timeout"
                continue
            if reason:
                self._cv_aborted = "CV" in reason; return False, reason
            g = get_geom(allow_stale=False)
            if g is not None: break
            if (time.monotonic() - t0) > self.servo_align_timeout_sec: return False, "VS no targets"
            time.sleep(min(0.15, dt))

        locked_quat = g["quat"] if self.servo_lock_orientation else None
        def quat_for_step(geom: dict): return locked_quat if locked_quat is not None else geom["quat"]

        # 1) Approach (Travel phase)
        self._set_obstacle_phase("travel", "VS approach")
        ok = self._plan_exec_pose(self._make_pose(g["ee_approach"], quat_for_step(g)), True, enforce_joint2_limit=True, context="VS approach")
        if not ok: return False, "Fail VS approach"
        cmd_pos = g["ee_approach"].copy()

        # 2) Align Hover (Press Phase)
        self._set_obstacle_phase("press", "VS align")
        if self._object_avoidance_enabled and self._table_collision_z_shift == 0.0:
            self._shift_table_collision_z(-0.04, "VS align")

        t_align = time.monotonic()
        settled = 0
        while True:
            reason = self._check_abort("VS align", cv_phase="align")
            if reason: self._cv_aborted = "CV" in reason; return False, reason
            g = get_geom(allow_stale=True)
            if g is None: return False, "Lost targets VS align"
            desired = g["ee_hover"]
            err = float(np.linalg.norm(desired - cmd_pos))
            if err <= self.servo_align_tol_m:
                settled += 1
                if settled >= 2: break
                time.sleep(min(0.15, dt)); continue
            settled = 0
            cmd_pos = self._step_toward(cmd_pos, desired, self.servo_step_m)
            if not self._plan_exec_pose(self._make_pose(cmd_pos, quat_for_step(g)), True, enforce_joint2_limit=True, context="VS align"): return False, "Fail align"
            if (time.monotonic() - t_align) > self.servo_align_timeout_sec: return False, "Align timeout"

        # 3) Press (Press Phase)
        t_press = time.monotonic()
        while True:
            reason = self._check_abort("VS press", cv_phase="press")
            if reason == WAIT_FOR_VISION:
                if not self._wait_for_vision_resume("VS press"): return False, "Vision unrecovered"
                continue
            if reason: self._cv_aborted = "CV" in reason; return False, reason
            g = get_geom(allow_stale=False)
            if g is None:
                if self.resume_then_ignore_vision_loss and self._vision_resumed_once and last_geom is not None: g = last_geom
                elif self.pause_on_vision_loss:
                    if not self._wait_for_vision_resume("VS press"): return False, "Vision unrecovered"
                    g = get_geom(allow_stale=True)
                    if g is None and last_geom is not None: g = last_geom
                    if g is None: return False, "No geom after resume"
                else: return False, "Lost target VS press"

            desired = g["ee_press"]
            if float(np.linalg.norm(desired - cmd_pos)) <= 0.003: break
            cmd_pos = self._step_toward(cmd_pos, desired, self.servo_press_step_m)
            if not self._plan_exec_pose(self._make_pose(cmd_pos, quat_for_step(g)), True, enforce_joint2_limit=False, context="VS press"): return False, "Fail press"
            if (time.monotonic() - t_press) > self.servo_press_timeout_sec: return False, "Press timeout"

        # 4) Retract (Press Phase)
        t_retract = time.monotonic()
        while True:
            g = get_geom(allow_stale=True)
            if g is None and last_geom is None: break
            if g is None: g = last_geom
            desired = g["ee_hover"]
            if float(np.linalg.norm(desired - cmd_pos)) <= self.servo_align_tol_m: break
            cmd_pos = self._step_toward(cmd_pos, desired, self.servo_step_m)
            if not self._plan_exec_pose(self._make_pose(cmd_pos, quat_for_step(g)), True, enforce_joint2_limit=True, context="VS retract"): return False, "Fail retract"
            if (time.monotonic() - t_retract) > self.servo_align_timeout_sec: break

        # 5) Home (Travel Phase)
        if self.servo_go_home_after_press:
            self._set_obstacle_phase("travel", "VS home")
            if self._object_avoidance_enabled and self._table_collision_z_shift != 0.0:
                self._restore_table_collision("VS home")
            if not self._plan_exec_named(self.get_parameter("home_named_target").value): return False, "Fail home"

        return True, "VS Complete"

    def _run_press_sequence(self):
        try:
            if self.press_mode == "visual_servo_white":
                ok, msg = self._run_press_visual_servo()
                self._press_result = (ok, msg)
                return

            with self._perception_lock:
                if self._paired_white_xyz is None or self._paired_gray_xyz is None:
                    self._press_result = (False, "No paired buttons!"); return
                p1, p2 = self._paired_white_xyz.copy(), self._paired_gray_xyz.copy()
            
            fwd = _compute_forward(p1, p2)
            z_hat = np.array([0.0, 0.0, 1.0])
            approach_m = float(self.get_parameter("approach_in").value) * INCH_TO_M
            hover_m = float(self.get_parameter("hover_in").value) * INCH_TO_M
            tip_offset_m = float(self.get_parameter("tip_offset_in").value) * INCH_TO_M

            def ee(tip): return tip - fwd * tip_offset_m
            quat = _make_orientation_quat(fwd, z_hat, self.get_parameter("tool_forward_axis").value, self.get_parameter("tool_up_axis").value)

            steps = [
                ("1) approach p2",     ee(p2 - fwd * approach_m + z_hat * hover_m), False, "travel"),
                ("2) hover p2",        ee(p2 + z_hat * hover_m),                    True,  "press"),
                ("3) press p2",        ee(p2.copy()),                               True,  "press"),
                ("4) return hover p2", ee(p2 + z_hat * hover_m),                    True,  "press"),
                ("5) transit -> p1",   ee(p1 + z_hat * hover_m),                    True,  "press"),
                ("6) press p1",        ee(p1.copy()),                               True,  "press"),
                ("7) return hover p1", ee(p1 + z_hat * hover_m),                    True,  "press"),
                ("8) retreat",         ee(p1 - fwd * approach_m + z_hat * hover_m), False, "travel"),
            ]

            for label, pos, linear, obs_mode in steps:
                is_press_step = label.startswith("3)") or label.startswith("6)")
                cv_phase = "press" if is_press_step else "align"

                self._set_obstacle_phase(obs_mode, f"snap:{label}")
                
                if self._object_avoidance_enabled:
                    if obs_mode == "press" and self._table_collision_z_shift == 0.0:
                        self._shift_table_collision_z(-0.04, label)
                    elif obs_mode == "travel" and self._table_collision_z_shift != 0.0:
                        self._restore_table_collision(label)

                reason = self._check_abort(label, cv_phase)
                if reason == WAIT_FOR_OBSTACLE:
                    if not self._wait_for_obstacle_clear(label):
                        self._obstacle_aborted = True; self._press_result = (False, f"Obstacle timeout: {label}"); return
                elif reason == WAIT_FOR_VISION:
                    if not self._wait_for_vision_resume(label):
                        self._press_result = (False, f"Vision timeout: {label}"); return
                elif reason:
                    self._cv_aborted = "CV" in reason; self._obstacle_aborted = "Obstacle" in reason; self._press_result = (False, reason); return

                ok = self._plan_exec_pose(self._make_pose(pos, quat), linear, enforce_joint2_limit=(cv_phase == "align"), context=label)
                if not ok: self._press_result = (False, f"Fail at {label}"); return

            self._set_obstacle_phase("travel", "snap:go_home")
            if self._object_avoidance_enabled: self._restore_table_collision("go_home")
            if not self._plan_exec_named(self.get_parameter("home_named_target").value):
                self._press_result = (False, "Fail go_home"); return

            self._press_result = (True, "Done")

        except Exception as e:
            self.get_logger().error(f"Thread exception: {e}")
            self._press_result = (False, f"Exception: {e}")
        finally:
            if self._object_avoidance_enabled: self._restore_table_collision("exit")

    def _tick(self):
        if self.state == PRE_SCAN:
            if self._stable_fan(self.prescan_dwell):
                self._fan_found_scan_idx = self.scan_idx
                self._enter_hold()
                return
            if self.prescan_cycles > self.prescan_max_cycles:
                self._enter_hold()
                return
            if self.scan_reached_mono is None:
                self._move_to_scan_pose("PRE_SCAN")
                return
            elapsed = time.monotonic() - self.scan_reached_mono
            if self.fan_ok_current:
                self._fan_found_scan_idx = self.scan_idx
                if elapsed < (self.prescan_settle * 2.0 + self.scan_extra_pause): return
            if elapsed < (self.prescan_settle + self.scan_extra_pause): return
            self.scan_idx = (self.scan_idx + 1) % len(SCAN_POSES)
            if self.scan_idx == 0: self.prescan_cycles += 1
            self.scan_reached_mono = None

        elif self.state == HOLD:
            if self.variant == "commit_only" and self.commit_only_single_shot and self._commit_only_attempted: return
            in_cap = self._hold_track_occlusion()
            if self.hold_timeout > 0 and self.variant == "hac" and self._hold_entered_mono and not self._execute_target_available() and (time.monotonic() - self._hold_entered_mono) > self.hold_timeout:
                self._enter_assist(start_scan_idx=self._fan_found_scan_idx or 0); return
            if in_cap: return
            if self.variant == "feas_only" and self._stable_feasible() and self._execute_target_available():
                self._start_execute(); return
            if self.auto_commit and not self._auto_committed and self.variant != "feas_only" and self._stable_fan(self.auto_commit_dwell) and self._execute_target_available():
                self._auto_committed = True; self._do_commit()

        elif self.state == ASSIST:
            if self.variant != "hac": self._enter_hold(); return
            if self.assist_cycles > self.assist_max_cycles: self._auto_committed = False; self._enter_hold(); return
            have_btn = self._execute_target_available(); st_cv = self._stable_feasible()
            if have_btn and (st_cv or self.hac_allow_latched_execute):
                self._fan_found_scan_idx = self.scan_idx; self._start_execute(); return
            if self.scan_reached_mono is None:
                self._move_to_scan_pose("ASSIST"); return
            elapsed = time.monotonic() - self.scan_reached_mono
            if self.fan_ok_current:
                self._fan_found_scan_idx = self.scan_idx
                if elapsed < (self.assist_settle * 3.0 + self.scan_extra_pause): return
            elif elapsed < (self.assist_settle + self.scan_extra_pause): return
            self.scan_idx = (self.scan_idx + 1) % len(SCAN_POSES)
            if self.scan_idx == 0: self.assist_cycles += 1
            self.scan_reached_mono = None

        elif self.state == EXECUTE:
            if self._press_thread is None: return
            if self._press_thread.is_alive():
                if self._press_deadline is not None and time.monotonic() > self._press_deadline + 15.0:
                    self._cv_aborted = False; self._obstacle_aborted = False; self._press_result = (False, "Hung thread")
                    self._press_thread = None; self._auto_committed = False; self._enter_hold()
                return

            self._press_thread = None
            success, message = self._press_result or (False, "No result")
            t_trial = self._t_start_mono - self.trial_t0
            self._set_obstacle_phase("travel", "done")
            if self._object_avoidance_enabled: self._restore_table_collision("done")

            if success:
                self._log_event(1, t_trial, self._trial_t(), OUTCOME_SUCCESS)
                self._auto_committed = False; self._enter_hold()
            elif self._obstacle_aborted:
                self._log_event(1, t_trial, float("nan"), OUTCOME_ABORT)
                self._auto_committed = False
                if self._execute_attempt_count == 1:
                    self._start_execute(is_retry=True); return
                if self.variant in ("hac", "feas_only"): self._enter_assist(start_scan_idx=self._fan_found_scan_idx or 0)
                else: self._enter_hold()
            elif self._cv_aborted:
                self._log_event(1, t_trial, float("nan"), OUTCOME_ABORT)
                self._auto_committed = False
                if self.variant == "hac": self._enter_assist(start_scan_idx=self._fan_found_scan_idx or 0)
                else: self._enter_hold()
            else:
                self._log_event(1, t_trial, float("nan"), OUTCOME_ABORT)
                self._auto_committed = False
                if self.variant in ("hac", "feas_only"): self._enter_assist(start_scan_idx=self._fan_found_scan_idx or 0)
                else: self._enter_hold()

def main():
    rclpy.init()
    node = FanPressSupervisor()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try: executor.spin()
    except KeyboardInterrupt: pass
    finally:
        try: node.destroy_node()
        except Exception: pass
        try: rclpy.shutdown()
        except Exception: pass

if __name__ == "__main__":
    main()