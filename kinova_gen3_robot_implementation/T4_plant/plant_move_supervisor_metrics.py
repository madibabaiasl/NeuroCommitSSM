#!/usr/bin/env python3
"""
plant_move_supervisor_metrics.py (ROS 2 Jazzy)

Adapted from fan_press_supervisor_metrics.py for the "Move Plant" task.

Task:
  1. Scan through 4 poses to locate the plant via Roboflow.
  2. Once found, move the end effector to a start point 5 inches LEFT of
     the plant (no grasp, just side contact).
  3. Sweep RIGHT slowly for 7 inches using Pilz LIN so the plant slides right.
  4. Return to rest pose.

Variants:
  - commit_only:
      Parse pred_csv, wait for first COMMIT (where gt_action==1), then execute.
      NO obstacle avoidance — if an obstacle is encountered, it is ignored.

  - feas_only:
      No pred_csv timing. Start when plant is stably detected.
      If an obstacle is detected during the forward stage => ABORT.

  - hold_commit:
      Wait for COMMIT time from pred_csv AND stable feasibility.
      If obstacle during the forward stage => PAUSE 5 seconds, then continue
      even if obstacle remains).

  - hac:
      Wait for COMMIT time from pred_csv AND stable feasibility.
      If obstacle during the forward stage => actively WAIT for removal,
      then continue.

Obstacle definition:
  - Depth reading closer than 2 inches (0.0508 m).
  - Obstacle checking is active continuously while moving FORWARD toward
    the plant (Step 4), then disabled once the robot reaches the left-aligned
    point before the push-right step.

Run example:
  ros2 run bottle_grasping plant_move_supervisor_metrics --ros-args \\
    -p variant:=commit_only \\
    -p trial_id:=1 \\
    -p pred_csv:=/home/pascal/Downloads/Task-3-test/Task_3/020_T3_pred_4.csv \\
    -p log_dir:=/home/pascal/Downloads/Task-3-test/Task_3/results/logs/001_T3_commit_only \\
    -p global_summary_csv:=/home/pascal/Downloads/Task-3-test/Task_3/plant_metrics_summary.csv \\
    -p max_runtime_s:=120.0 \\
    -p stop_after_success:=true
"""

import os
import csv
import math
import time
import threading
from copy import deepcopy
from collections import deque
from typing import Optional, List, Tuple, Dict, Any, Callable

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped, Pose, TwistStamped
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

from action_msgs.msg import GoalStatus
from control_msgs.action import GripperCommand
from moveit_msgs.msg import MoveItErrorCodes, CollisionObject as CollisionObjectMsg
from shape_msgs.msg import SolidPrimitive
from moveit_configs_utils import MoveItConfigsBuilder
from moveit.planning import MoveItPy, PlanRequestParameters
from moveit.core.robot_state import RobotState


INCH_TO_M = 0.0254

# ─── Supervisor states ───────────────────────────────────────────────
PRE_SCAN = "PRE_SCAN"
HOLD     = "HOLD"
ASSIST   = "ASSIST"
EXECUTE  = "EXECUTE"

VARIANT_ID = {"commit_only": 0, "feas_only": 1, "hold_commit": 2, "hac": 3}

# ─── Quality topic indices (from plant_pose_node_metrics) ────────────
Q_PLANT_CONF   = 0
Q_POSE_MODE    = 1
Q_TF_OK        = 2
Q_PLANT_VR     = 3
Q_PLANT_BBOX   = 4
Q_PLANT_DEPTH  = 5
Q_PLANT_USED   = 6
Q_MIN_LEN      = 7

# ─── Scan poses (same 4 poses as fan task — adjust to your workspace) ─
SCAN_POSES: List[List[float]] = [
    [1.5701, 0.7471, 1.3472, 0.1524, 1.2910, -1.6665],
    [1.2839, 0.8440, 1.3817, 0.2653, 1.4562, -1.6168],
    [0.6427, 0.9756, 1.4850, 0.2295, 1.2395, -1.6416],
    [0.5322, 0.8251, 1.2608, 0.1999, 1.6878, -1.6803],
]

# ─── Robot speed ─────────────────────────────────────────────────────
DEFAULT_MAX_VEL = 0.30
DEFAULT_MAX_ACC = 0.30
PILZ_LIN_VEL   = 0.05   # slow linear for sweep
PILZ_LIN_ACC   = 0.05

# ─── Obstacle thresholds ────────────────────────────────────────────
OBSTACLE_STOP_DISTANCE_M  = 8.0 * INCH_TO_M   # 2 inches = 0.0508 m
OBSTACLE_RESUME_DISTANCE_M = 3.0 * INCH_TO_M  # resume hysteresis


# =====================================================================
# Utility functions
# =====================================================================

def _safe_float(x, default=float("nan")):
    try:
        return float(x)
    except Exception:
        return default


def _count_flaps(commit_states: List[str]) -> int:
    if not commit_states:
        return 0
    flaps = 0
    prev = commit_states[0]
    for s in commit_states[1:]:
        if s != prev:
            flaps += 1
            prev = s
    return flaps


def _read_model_csv(pred_csv: str) -> Optional[dict]:
    if pred_csv is None or str(pred_csv).strip() == "":
        return None
    pred_csv = str(pred_csv)
    if not os.path.exists(pred_csv):
        raise FileNotFoundError(f"pred_csv not found: {pred_csv}")

    with open(pred_csv, "r", newline="") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
    if not rows:
        raise ValueError(f"pred_csv is empty: {pred_csv}")

    def col(name, cast):
        out = []
        for r in rows:
            out.append(cast(r.get(name)))
        return np.asarray(out)

    model = {
        "center_time_s": col("center_time_s",
                             lambda v: _safe_float(v)),
        "commit_event":  col("commit_event",
                             lambda v: int(float(_safe_float(v, 0.0)))),
        "commit_state":  np.asarray(
            [str(r.get("commit_state", "")).strip().upper() for r in rows]),
        "gt_action":     col("gt_action",
                             lambda v: int(float(_safe_float(v, 0.0)))),
    }
    return model


def _first_commit_time(model: Optional[dict]) -> Optional[float]:
    """Return center_time_s of first COMMIT row where gt_action == 1."""
    if model is None:
        return None

    cs = [str(x).strip().upper() for x in model.get("commit_state", [])]
    gt = model.get("gt_action", np.array([]))

    for i, s in enumerate(cs):
        if s == "COMMIT" and i < len(gt) and int(gt[i]) == 1:
            return float(model["center_time_s"][i])

    ce = model.get("commit_event", None)
    if ce is not None:
        for i, e in enumerate(ce):
            if int(e) == 1 and i < len(gt) and int(gt[i]) == 1:
                return float(model["center_time_s"][i])

    return None


def _nearest_index(times: np.ndarray, t: float) -> int:
    return int(np.argmin(np.abs(times - float(t))))


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
    """Build a quaternion that points *tool_forward_axis* along
    *forward_world* and *tool_up_axis* along *up_world*."""
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


def _augment_moveit_cfg(cfg: dict, jst: str = "/joint_states") -> dict:
    cfg = dict(deepcopy(cfg))

    pp = cfg.get("planning_pipelines", None)
    if pp is None:
        cfg["planning_pipelines"] = {"pipeline_names": ["ompl"]}
    elif isinstance(pp, list):
        cfg["planning_pipelines"] = {"pipeline_names": pp if pp else ["ompl"]}
    elif isinstance(pp, dict):
        if "pipeline_names" not in pp or not pp["pipeline_names"]:
            pp["pipeline_names"] = ["ompl"]
        cfg["planning_pipelines"] = pp

    if "default_planning_pipeline" not in cfg:
        names = cfg["planning_pipelines"].get("pipeline_names", ["ompl"])
        cfg["default_planning_pipeline"] = names[0] if names else "ompl"

    cfg.setdefault(
        "planning_scene_monitor_options",
        {
            "name": "planning_scene_monitor",
            "robot_description": "robot_description",
            "joint_state_topic": jst,
            "attached_collision_object_topic": "/attached_collision_object",
            "publish_planning_scene_topic": "/moveit_py/publish_planning_scene",
            "monitored_planning_scene_topic": "/moveit_py/monitored_planning_scene",
            "wait_for_initial_state_timeout": 10.0,
        },
    )

    cfg.setdefault(
        "plan_request_params",
        {
            "planning_attempts": 1,
            "planning_pipeline": cfg.get("default_planning_pipeline", "ompl"),
            "max_velocity_scaling_factor": DEFAULT_MAX_VEL,
            "max_acceleration_scaling_factor": DEFAULT_MAX_ACC,
            "planning_time": 2.5,
        },
    )
    return cfg


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


# =====================================================================
# Supervisor Node
# =====================================================================

class PlantMoveSupervisor(Node):
    def __init__(self):
        super().__init__("plant_move_supervisor")

        self._sub_cg = ReentrantCallbackGroup()
        self._srv_cg = MutuallyExclusiveCallbackGroup()

        # ── Variant ─────────────────────────────────────────────────
        self.declare_parameter("variant", "feas_only")
        self.variant = str(self.get_parameter("variant").value).strip()
        if self.variant not in VARIANT_ID:
            raise ValueError(f"Unknown variant '{self.variant}'")

        self.declare_parameter("trial_id", 0)
        self.trial_id = int(self.get_parameter("trial_id").value)
        self.trial_t0 = time.monotonic()

        self.declare_parameter("pred_csv", "")
        self.declare_parameter("log_dir", "/tmp/plant_logs")
        self.declare_parameter(
            "global_summary_csv", "/tmp/plant_metrics_summary.csv")
        self.declare_parameter("max_runtime_s", 120.0)
        self.declare_parameter("stop_after_success", True)
        self.declare_parameter("stop_after_failure", True)
        self.declare_parameter("max_execute_attempts", 1)
        self.declare_parameter("retry_cooldown_s", 2.0)

        self.pred_csv = str(self.get_parameter("pred_csv").value)
        self.log_dir = str(self.get_parameter("log_dir").value)
        self.global_summary_csv = str(
            self.get_parameter("global_summary_csv").value)
        self.max_runtime_s = float(self.get_parameter("max_runtime_s").value)
        self.stop_after_success = bool(
            self.get_parameter("stop_after_success").value)
        self.stop_after_failure = bool(
            self.get_parameter("stop_after_failure").value)
        self.max_execute_attempts = max(
            1, int(self.get_parameter("max_execute_attempts").value))
        self.retry_cooldown_s = max(
            0.0, float(self.get_parameter("retry_cooldown_s").value))

        os.makedirs(self.log_dir, exist_ok=True)
        self.state_path = os.path.join(self.log_dir, "state_stream.csv")
        self.cv_path = os.path.join(self.log_dir, "cv_stream.csv")
        self.attempts_path = os.path.join(self.log_dir, "attempts.csv")
        self.metrics_trial_path = os.path.join(
            self.log_dir, "metrics_trial.csv")

        self._init_csv(self.state_path,
                       ["t_s", "trial_id", "variant", "sup_state"])
        self._init_csv(self.cv_path,
                       ["t_s", "trial_id", "variant", "f_cv"])
        self._init_csv(self.attempts_path,
                       ["trial_id", "variant", "started_flag",
                        "t_start_s", "t_success_s", "outcome"])
        self._init_csv(self.metrics_trial_path, [
            "trial_id", "variant", "started_flag", "outcome",
            "t_start_s", "t_success_s", "time_to_success_s",
            "false_start", "gt_action_at_start",
            "cv_infeas_start", "f_cv_at_start",
            "flaps"
        ])

        self.global_summary_header = [
            "trial_id", "variant", "log_dir", "pred_csv",
            "started_flag", "outcome",
            "t_start_s", "t_success_s",
            "false_start", "cv_infeas_start",
            "flaps", "time_to_success_s",
        ]
        self._init_global_summary(
            self.global_summary_csv, self.global_summary_header)

        self.started_flag = 0
        self.t_start_s = float("nan")
        self.t_success_s = float("nan")
        self.outcome = "timeout"
        self._finished = False
        self._final_saved = False
        self._attempt_idx = 0
        self._attempt_active = False
        self._attempt_start_s = float("nan")
        self._next_retry_mono = 0.0
        self._max_attempts_exceeded_logged = False

        # ── Load pred CSV ───────────────────────────────────────────
        self.model: Optional[dict] = None
        self.t_commit_s: Optional[float] = None
        needs_commit = self.variant in ("commit_only", "hold_commit", "hac")
        try:
            if needs_commit and self.pred_csv.strip():
                self.model = _read_model_csv(self.pred_csv)
                self.t_commit_s = _first_commit_time(self.model)
                if self.t_commit_s is not None:
                    self.get_logger().info(
                        f"[MODEL] first commit (gt==1) at "
                        f"t={self.t_commit_s:.3f}s from {self.pred_csv}")
                else:
                    self.get_logger().warn(
                        f"No commit_event==1 with gt_action==1 "
                        f"found in {self.pred_csv}")
        except Exception as e:
            self.get_logger().error(f"Failed to load pred_csv: {e}")
            self.model = None
            self.t_commit_s = None

        # ── Pre-scan / perception params ────────────────────────────
        self.declare_parameter("enable_prescan", True)
        self.declare_parameter("min_plant_conf", 0.50)
        self.declare_parameter("prescan_dwell_sec", 0.40)
        self.declare_parameter("prescan_pose_settle_sec", 2.2)
        self.declare_parameter("scan_extra_pause_sec", 0.6)
        self.declare_parameter("prescan_max_cycles", 1000000)

        self.enable_prescan = bool(
            self.get_parameter("enable_prescan").value)
        self.min_plant_conf = float(
            self.get_parameter("min_plant_conf").value)
        self.prescan_dwell = float(
            self.get_parameter("prescan_dwell_sec").value)
        self.prescan_settle = float(
            self.get_parameter("prescan_pose_settle_sec").value)
        self.scan_extra_pause = float(
            self.get_parameter("scan_extra_pause_sec").value)
        self.prescan_max_cycles = int(
            self.get_parameter("prescan_max_cycles").value)

        # Feasibility checks on plant detection quality
        self.declare_parameter("min_used_px", 10.0)
        self.declare_parameter("min_valid_ratio", 0.20)
        self.declare_parameter("dwell_sec", 0.60)
        self.declare_parameter("plant_stale_sec", 8.0)

        self.min_used_px = float(self.get_parameter("min_used_px").value)
        self.min_valid_ratio = float(
            self.get_parameter("min_valid_ratio").value)
        self.dwell_sec = float(self.get_parameter("dwell_sec").value)
        self._plant_stale_sec = float(
            self.get_parameter("plant_stale_sec").value)

        # ── Motion params ───────────────────────────────────────────
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("ee_link", "end_effector_link")
        self.declare_parameter("tool_forward_axis", "z")
        self.declare_parameter("tool_up_axis", "y")

        # Sweep geometry — simple L-shaped push
        self.declare_parameter("left_offset_in", 5.0)       # go 5in left of plant
        self.declare_parameter("straight_distance_in", 15.0) # go straight 15in toward plant
        self.declare_parameter("sweep_right_in", 8.0)        # push right 5in
        self.declare_parameter("hover_in", 2.0)              # hover height above plant
        self.declare_parameter("approach_back_in", 5.0)      # extra distance behind for safe approach
        self.declare_parameter("tip_offset_in", 6.0)         # EE link to tool tip
        self.declare_parameter("home_named_target", "home")
        self.declare_parameter("try_pilz_lin", True)
        self.declare_parameter("pilz_namespace", "pilz_lin")
        self.declare_parameter("sweep_pilz_vel", 0.05)
        self.declare_parameter("sweep_pilz_acc", 0.05)
        self.declare_parameter("straight_use_cartesian_vector", True)
        self.declare_parameter("straight_vector_topic", "/servo_node/delta_twist_cmds")
        self.declare_parameter("straight_vector_backend", "auto")  # auto|servo|lin_step
        self.declare_parameter("straight_vector_frame", "base_link")
        self.declare_parameter("straight_vector_speed_mps", 0.03)
        self.declare_parameter("straight_vector_hz", 30.0)
        self.declare_parameter("straight_vector_step_in", 1.0)  # for lin_step fallback
        self.declare_parameter("straight_vector_require_subscriber", True)
        self.declare_parameter("straight_vector_enable_servo", True)
        self.declare_parameter("servo_start_service", "/servo_node/start_servo")
        self.declare_parameter("debug_motion_targets", True)
        self.declare_parameter(
            "gripper_action_name", "/robotiq_gripper_controller/gripper_cmd")
        self.declare_parameter("gripper_open_position", 0.0)
        self.declare_parameter("gripper_close_deg", 13.0)
        self.declare_parameter("gripper_max_effort", 50.0)
        self.declare_parameter("gripper_command_timeout_s", 4.0)
        self.declare_parameter("gripper_close_before_sweep", False)
        self.declare_parameter("gripper_open_after_sweep", False)

        self.declare_parameter(
            "robot_name", "kinova_gen3_6dof_robotiq_2f_85")
        self.declare_parameter(
            "moveit_config_pkg",
            "kinova_gen3_6dof_robotiq_2f_85_moveit_config")
        self.declare_parameter("move_group_name", "manipulator")

        # ── Collision objects (table + wheelchair/chair + x-wall) ────
        self.declare_parameter("collision_enabled", True)
        self.declare_parameter("table_point", [-0.5156, 0.5, 0.18645])
        self.declare_parameter("table_yaw", math.pi / 2.0)
        self.declare_parameter("wheelchair_point", [0.0, -0.39, 0.12065])
        self.declare_parameter("wheelchair_yaw", 0.0)
        self.declare_parameter("x_wall_distance_in", 30.0)
        self.declare_parameter("x_wall_thickness_m", 0.03)
        self.declare_parameter("x_wall_width_y_m", 2.0)
        self.declare_parameter("x_wall_height_m", 1.2)

        self.declare_parameter("max_velocity_scaling", DEFAULT_MAX_VEL)
        self.declare_parameter("max_acceleration_scaling", DEFAULT_MAX_ACC)
        self.max_vel = float(
            self.get_parameter("max_velocity_scaling").value)
        self.max_acc = float(
            self.get_parameter("max_acceleration_scaling").value)
        self.sweep_pilz_vel = float(
            self.get_parameter("sweep_pilz_vel").value)
        self.sweep_pilz_acc = float(
            self.get_parameter("sweep_pilz_acc").value)
        self.straight_use_cartesian_vector = bool(
            self.get_parameter("straight_use_cartesian_vector").value)
        self.straight_vector_topic = str(
            self.get_parameter("straight_vector_topic").value)
        self.straight_vector_backend = str(
            self.get_parameter("straight_vector_backend").value).strip().lower()
        self.straight_vector_frame = str(
            self.get_parameter("straight_vector_frame").value)
        self.straight_vector_speed_mps = float(
            self.get_parameter("straight_vector_speed_mps").value)
        self.straight_vector_hz = float(
            self.get_parameter("straight_vector_hz").value)
        self.straight_vector_step_m = max(
            0.10 * INCH_TO_M,
            float(self.get_parameter("straight_vector_step_in").value) * INCH_TO_M)
        self.straight_vector_require_subscriber = bool(
            self.get_parameter("straight_vector_require_subscriber").value)
        self.straight_vector_enable_servo = bool(
            self.get_parameter("straight_vector_enable_servo").value)
        self.servo_start_service = str(
            self.get_parameter("servo_start_service").value)
        self.debug_motion_targets = bool(
            self.get_parameter("debug_motion_targets").value)
        self.gripper_action_name = str(
            self.get_parameter("gripper_action_name").value)
        self.gripper_open_position = float(
            self.get_parameter("gripper_open_position").value)
        self.gripper_close_deg = float(
            self.get_parameter("gripper_close_deg").value)
        self.gripper_close_position = float(math.radians(self.gripper_close_deg))
        self.gripper_max_effort = float(
            self.get_parameter("gripper_max_effort").value)
        self.gripper_command_timeout_s = float(
            self.get_parameter("gripper_command_timeout_s").value)
        self.gripper_close_before_sweep = bool(
            self.get_parameter("gripper_close_before_sweep").value)
        self.gripper_open_after_sweep = bool(
            self.get_parameter("gripper_open_after_sweep").value)
        self._gripper_client = ActionClient(
            self, GripperCommand, self.gripper_action_name)
        self._straight_twist_pub = self.create_publisher(
            TwistStamped, self.straight_vector_topic, 10)
        self._servo_start_cli = self.create_client(
            Trigger, self.servo_start_service)
        self._servo_enabled = False
        if self.straight_vector_backend not in ("auto", "servo", "lin_step"):
            self.get_logger().warn(
                f"Unknown straight_vector_backend={self.straight_vector_backend}; using auto")
            self.straight_vector_backend = "auto"

        # ── Obstacle gate ───────────────────────────────────────────
        # For plant task: obstacle = depth < 2 inches
        # Activated only during forward stage (Step 4), not during scan/push.
        self.declare_parameter("obstacle_monitor_enabled", True)
        self.declare_parameter(
            "obstacle_depth_topic", "/camera/depth_registered/image_rect")
        self.declare_parameter(
            "obstacle_info_topic", "/camera/color/camera_info")
        self.declare_parameter("obstacle_roi_fraction", 0.30)
        self.declare_parameter("obstacle_min_valid_pixels", 80)
        self.declare_parameter("obstacle_percentile", 5.0)
        self.declare_parameter("obstacle_median_window", 5)
        self.declare_parameter("obstacle_stop_confirm_frames", 2)
        self.declare_parameter("obstacle_resume_confirm_frames", 3)
        self.declare_parameter("obstacle_depth_timeout_sec", 1.0)
        self.declare_parameter("obstacle_fail_safe_on_no_depth", True)
        self.declare_parameter("obstacle_debug_period_s", 0.20)

        self.obstacle_monitor_enabled = bool(
            self.get_parameter("obstacle_monitor_enabled").value)
        self.obstacle_depth_topic = str(
            self.get_parameter("obstacle_depth_topic").value)
        self.obstacle_info_topic = str(
            self.get_parameter("obstacle_info_topic").value)
        self.obstacle_roi_fraction = float(
            self.get_parameter("obstacle_roi_fraction").value)
        self.obstacle_min_valid_pixels = int(
            self.get_parameter("obstacle_min_valid_pixels").value)
        self.obstacle_percentile = float(
            self.get_parameter("obstacle_percentile").value)
        self.obstacle_median_window = int(
            self.get_parameter("obstacle_median_window").value)
        self.obstacle_stop_confirm_frames = int(
            self.get_parameter("obstacle_stop_confirm_frames").value)
        self.obstacle_resume_confirm_frames = int(
            self.get_parameter("obstacle_resume_confirm_frames").value)
        self.obstacle_depth_timeout_sec = float(
            self.get_parameter("obstacle_depth_timeout_sec").value)
        self.obstacle_fail_safe_on_no_depth = bool(
            self.get_parameter("obstacle_fail_safe_on_no_depth").value)
        self.obstacle_debug_period_s = max(
            0.0, float(self.get_parameter("obstacle_debug_period_s").value))

        self.obstacle_stop_distance_m = OBSTACLE_STOP_DISTANCE_M
        self.obstacle_resume_distance_m = OBSTACLE_RESUME_DISTANCE_M

        # Obstacle gate is enabled for feas_only, hold_commit, hac
        # commit_only never uses obstacle avoidance
        self._obstacle_gate_enabled = (
            self.obstacle_monitor_enabled
            and self.variant in ("feas_only", "hold_commit", "hac")
        )
        # Gate is NOT active until sweep phase starts
        self._obstacle_gate_active = False

        # ── Perception state ────────────────────────────────────────
        self._perception_lock = threading.Lock()
        self.last_quality: Optional[list] = None

        self.plant_ok_current = False
        self.plant_ok_true_since = None

        self.f_cv_current = False
        self.f_cv_true_since = None

        self._plant_xyz: Optional[np.ndarray] = None
        self._plant_stamp: Optional[float] = None

        # ── Obstacle state ──────────────────────────────────────────
        self._obstacle_lock = threading.Lock()
        self._obstacle_bridge = (
            CvBridge() if self._obstacle_gate_enabled else None)
        self._obstacle_last_depth_mono = time.monotonic()
        self._obstacle_closest_hist = deque(
            maxlen=max(1, self.obstacle_median_window))
        self._obstacle_closest_dist = float("inf")
        self._obstacle_blocked = False
        self._obstacle_stop_hits = 0
        self._obstacle_clear_hits = 0
        self._obstacle_timeout_active = False
        self._obstacle_min_depth_m = float("inf")
        self._obstacle_last_debug_log_mono = 0.0
        self._scene_collision_lock = threading.Lock()

        # ── State machine ───────────────────────────────────────────
        self.state = PRE_SCAN if self.enable_prescan else HOLD
        self.scan_idx = 0
        self.scan_reached_mono = None
        self.prescan_cycles = 0
        self._plant_found_scan_idx: Optional[int] = None

        self._sweep_thread: Optional[threading.Thread] = None
        self._sweep_result: Optional[Tuple[bool, str]] = None
        self._t_start_mono: Optional[float] = None

        self.declare_parameter("hold_commit_obstacle_pause_sec", 5.0)
        self.hold_commit_obstacle_pause_sec = float(
            self.get_parameter("hold_commit_obstacle_pause_sec").value)

        # ── Subscriptions ───────────────────────────────────────────
        self.create_subscription(
            Float32MultiArray, "/plant/quality",
            self._on_quality, 10, callback_group=self._sub_cg)
        self.create_subscription(
            PoseStamped, "/plant/pose_base",
            self._on_plant_pose, 10, callback_group=self._sub_cg)

        if self._obstacle_gate_enabled:
            self.create_subscription(
                Image, self.obstacle_depth_topic,
                self._on_obstacle_depth, 10, callback_group=self._sub_cg)
            self.create_subscription(
                CameraInfo, self.obstacle_info_topic,
                self._on_obstacle_info, 10, callback_group=self._sub_cg)
            self.get_logger().info(
                f"Obstacle gate enabled for variant={self.variant} "
                f"stop={self.obstacle_stop_distance_m:.4f}m "
                f"({self.obstacle_stop_distance_m / INCH_TO_M:.1f}in)")
        else:
            self.get_logger().info(
                f"Obstacle gate DISABLED for variant={self.variant}")

        self.create_service(
            Trigger, "/commit_plant_move",
            self._on_commit, callback_group=self._srv_cg)

        # ── MoveIt init ─────────────────────────────────────────────
        self._moveit = None
        self._pc = None
        self._pilz_params = None
        self._pilz_namespace = str(
            self.get_parameter("pilz_namespace").value)
        self._default_planning_pipeline = "ompl"
        self._group_name = str(
            self.get_parameter("move_group_name").value)

        rn = str(self.get_parameter("robot_name").value)
        pkg = str(self.get_parameter("moveit_config_pkg").value)

        try:
            mcfg = MoveItConfigsBuilder(
                rn, package_name=pkg).to_moveit_configs()
            cd = _augment_moveit_cfg(
                mcfg.to_dict(), jst="/joint_states")
            self._default_planning_pipeline = str(
                cd.get("default_planning_pipeline", "ompl"))

            self._moveit = MoveItPy(
                node_name="plant_sup_moveitpy", config_dict=cd)
            self._pc = self._moveit.get_planning_component(
                self._group_name)
            self.get_logger().info(
                f"MoveItPy ready ({rn} / {self._group_name}) "
                f"default_pipeline={self._default_planning_pipeline} "
                f"vel={self.max_vel:.0%} acc={self.max_acc:.0%}")

            if bool(self.get_parameter("try_pilz_lin").value):
                ns = self._pilz_namespace
                try:
                    self._pilz_params = PlanRequestParameters(
                        self._moveit, ns)
                    self._pilz_params.planning_pipeline = (
                        "pilz_industrial_motion_planner")
                    self._pilz_params.planner_id = "LIN"
                    self._pilz_params.planning_time = 2.0
                    self._pilz_params.planning_attempts = 1
                    self._pilz_params.max_velocity_scaling_factor = (
                        PILZ_LIN_VEL)
                    self._pilz_params.max_acceleration_scaling_factor = (
                        PILZ_LIN_ACC)
                    self.get_logger().info(
                        f"Pilz LIN enabled (ns={ns}) "
                        f"vel={PILZ_LIN_VEL} acc={PILZ_LIN_ACC}")
                except Exception as e:
                    self.get_logger().warn(f"Pilz init failed: {e}")

        except Exception as e:
            self.get_logger().error(f"MoveItPy init failed: {e}")
            self._moveit = None
            self._pc = None

        if self._moveit is None or self._pc is None:
            self.get_logger().error(
                "MoveIt is not initialized -> shutting down.")
            self._finished = True
            self.create_timer(0.1, self._shutdown_soon)
            return

        self._collision_refresh_timer = None
        if bool(self.get_parameter("collision_enabled").value):
            self._setup_collision_objects(log=True, blocking=True)
            self._collision_refresh_timer = self.create_timer(
                0.2, self._refresh_collision_objects,
                callback_group=self._sub_cg)

        self.create_timer(0.05, self._tick)
        self.create_timer(0.25, self._watchdog_timeout)

        self.get_logger().info(
            f"PlantMoveSupervisor up — variant={self.variant} "
            f"state={self.state} trial={self.trial_id} "
            f"t_commit_s={self.t_commit_s} "
            f"global_summary_csv={self.global_summary_csv}")

    # ─────────────────────────────────────────────────────────────
    # CSV helpers
    # ─────────────────────────────────────────────────────────────

    def _init_csv(self, path: str, header: List[str]) -> None:
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(header)

    def _init_global_summary(self, path: str, header: List[str]) -> None:
        _ensure_parent_dir(path)
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(header)

    def _append_csv(self, path: str, row: list):
        try:
            with open(path, "a", newline="") as f:
                csv.writer(f).writerow(row)
        except Exception:
            pass

    def _shutdown_soon(self):
        if not self._finished:
            return
        self._cleanup_motion_resources()
        try:
            self.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass

    def _cleanup_motion_resources(self):
        try:
            self._pc = None
        except Exception:
            pass
        try:
            self._pilz_params = None
        except Exception:
            pass
        try:
            self._moveit = None
        except Exception:
            pass

    def _request_shutdown(self):
        self._finished = True
        self._cleanup_motion_resources()
        try:
            self.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass

    def _trial_t(self) -> float:
        return time.monotonic() - self.trial_t0

    # ─────────────────────────────────────────────────────────────
    # Quaternion / geometry helpers
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _quat_multiply(a, b):
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        x = aw * bx + ax * bw + ay * bz - az * by
        y = aw * by - ax * bz + ay * bw + az * bx
        z = aw * bz + ax * by - ay * bx + az * bw
        w = aw * bw - ax * bx - ay * by - az * bz
        return (x, y, z, w)

    @staticmethod
    def _rotate_vec_by_quat(v, q):
        vx, vy, vz = v
        qx, qy, qz, qw = q
        ix = qw * vx + qy * vz - qz * vy
        iy = qw * vy + qz * vx - qx * vz
        iz = qw * vz + qx * vy - qy * vx
        iw = -qx * vx - qy * vy - qz * vz
        rx = ix * qw + iw * -qx + iy * -qz - iz * -qy
        ry = iy * qw + iw * -qy + iz * -qx - ix * -qz
        rz = iz * qw + iw * -qz + ix * -qy - iy * -qx
        return (rx, ry, rz)

    def _apply_base_to_local(self, base_pose: Pose,
                             local_pose: Pose) -> Pose:
        out = Pose()
        bq = (
            base_pose.orientation.x,
            base_pose.orientation.y,
            base_pose.orientation.z,
            base_pose.orientation.w,
        )
        local_pos = (
            local_pose.position.x,
            local_pose.position.y,
            local_pose.position.z,
        )
        rotated = self._rotate_vec_by_quat(local_pos, bq)
        out.position.x = base_pose.position.x + rotated[0]
        out.position.y = base_pose.position.y + rotated[1]
        out.position.z = base_pose.position.z + rotated[2]
        lq = (
            local_pose.orientation.x,
            local_pose.orientation.y,
            local_pose.orientation.z,
            local_pose.orientation.w,
        )
        rx, ry, rz, rw = self._quat_multiply(bq, lq)
        out.orientation.x = rx
        out.orientation.y = ry
        out.orientation.z = rz
        out.orientation.w = rw
        return out

    @staticmethod
    def _make_box(size: List[float]) -> SolidPrimitive:
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = list(size)
        return box

    @staticmethod
    def _make_collision_pose(position, yaw: float = 0.0) -> Pose:
        p = Pose()
        p.position.x, p.position.y, p.position.z = position
        half = yaw / 2.0
        p.orientation.x = 0.0
        p.orientation.y = 0.0
        p.orientation.z = math.sin(half)
        p.orientation.w = math.cos(half)
        return p

    # ─────────────────────────────────────────────────────────────
    # Collision objects (identical to fan task)
    # ─────────────────────────────────────────────────────────────

    def _build_collision_objects(self) -> Tuple[
            CollisionObjectMsg, CollisionObjectMsg, CollisionObjectMsg]:
        table_point = list(self.get_parameter("table_point").value)
        table_yaw = float(self.get_parameter("table_yaw").value)
        wheelchair_point = list(
            self.get_parameter("wheelchair_point").value)
        wheelchair_yaw = float(self.get_parameter("wheelchair_yaw").value)
        x_wall_distance_in = float(
            self.get_parameter("x_wall_distance_in").value)
        x_wall_thickness_m = float(
            self.get_parameter("x_wall_thickness_m").value)
        x_wall_width_y_m = float(
            self.get_parameter("x_wall_width_y_m").value)
        x_wall_height_m = float(
            self.get_parameter("x_wall_height_m").value)

        base_table_pose = self._make_collision_pose(table_point, table_yaw)
        base_wheel_pose = self._make_collision_pose(
            wheelchair_point, wheelchair_yaw)

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
        shelf_heights = [
            bottom_to_shelf1 - half_shelf_thickness,
            bottom_to_shelf2 - half_shelf_thickness,
            bottom_to_top - half_shelf_thickness,
        ]
        shelf1_to_shelf2 = bottom_to_shelf2 - bottom_to_shelf1
        shelf2_to_top = bottom_to_top - bottom_to_shelf2
        left_to_part1 = 0.37465

        table_ref = (
            table_abs_pose.position.x,
            table_abs_pose.position.y,
            table_abs_pose.position.z,
        )
        table_prims: List[Tuple[SolidPrimitive, Pose]] = []
        table_prims.append((self._make_box(table_size), table_abs_pose))

        for z_rel in shelf_heights:
            z = shelf_origin_z + z_rel + shelf_thickness / 2.0
            table_prims.append((
                self._make_box([
                    shelf_outer_size[0], shelf_outer_size[1],
                    shelf_thickness]),
                self._make_collision_pose((
                    shelf_origin_x + shelf_outer_size[0] / 2.0, 0.0, z)),
            ))

        table_prims.append((
            self._make_box([
                shelf_outer_size[0], 0.02, shelf_outer_size[2]]),
            self._make_collision_pose((
                shelf_origin_x + shelf_outer_size[0] / 2.0,
                shelf_origin_y - shelf_outer_size[1] / 2.0,
                shelf_origin_z + shelf_outer_size[2] / 2.0)),
        ))
        table_prims.append((
            self._make_box([
                shelf_outer_size[0], 0.02, shelf_outer_size[2]]),
            self._make_collision_pose((
                shelf_origin_x + shelf_outer_size[0] / 2.0,
                shelf_origin_y + shelf_outer_size[1] / 2.0,
                shelf_origin_z + shelf_outer_size[2] / 2.0)),
        ))
        table_prims.append((
            self._make_box([
                0.02, shelf_outer_size[1], shelf_outer_size[2]]),
            self._make_collision_pose((
                shelf_origin_x + shelf_outer_size[0] - 0.01,
                0.0,
                shelf_origin_z + shelf_outer_size[2] / 2.0)),
        ))
        table_prims.append((
            self._make_box([
                0.02, shelf_outer_size[1],
                bottom_to_shelf1 - half_shelf_thickness]),
            self._make_collision_pose((
                shelf_origin_x, 0.0,
                shelf_origin_z + bottom_to_shelf1 / 2.0)),
        ))
        middle_to_part = (shelf_outer_size[1] / 2.0) - left_to_part1
        table_prims.append((
            self._make_box([
                shelf_outer_size[0], 0.02, shelf1_to_shelf2]),
            self._make_collision_pose((
                shelf_origin_x + shelf_outer_size[0] / 2.0,
                middle_to_part,
                shelf_origin_z + shelf1_to_shelf2)),
        ))
        table_prims.append((
            self._make_box([
                shelf_outer_size[0], 0.02, shelf1_to_shelf2]),
            self._make_collision_pose((
                shelf_origin_x + shelf_outer_size[0] / 2.0,
                -middle_to_part,
                shelf_origin_z + shelf1_to_shelf2)),
        ))
        table_prims.append((
            self._make_box([
                shelf_outer_size[0], 0.02, shelf2_to_top]),
            self._make_collision_pose((
                shelf_origin_x + shelf_outer_size[0] / 2.0,
                0.0,
                shelf_origin_z + bottom_to_shelf1
                + shelf2_to_top - shelf_thickness / 2.0)),
        ))

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

        wall_size = [0.51, 0.5, 0.2413]
        wall_abs_pose = self._make_collision_pose(
            (0.0, -0.39, wall_size[2] / 2.0))
        wheel_ref = (
            wall_abs_pose.position.x,
            wall_abs_pose.position.y,
            wall_abs_pose.position.z,
        )

        co_wheel = CollisionObjectMsg()
        co_wheel.id = "wheelchair"
        co_wheel.header.frame_id = "base_link"
        co_wheel.operation = CollisionObjectMsg.ADD

        local = Pose()
        local.position.x = wall_abs_pose.position.x - wheel_ref[0]
        local.position.y = wall_abs_pose.position.y - wheel_ref[1]
        local.position.z = wall_abs_pose.position.z - wheel_ref[2]
        local.orientation = wall_abs_pose.orientation
        world_pose = self._apply_base_to_local(base_wheel_pose, local)
        co_wheel.primitives.append(self._make_box(wall_size))
        co_wheel.primitive_poses.append(world_pose)

        x_wall_x_m = x_wall_distance_in * INCH_TO_M
        co_xwall = CollisionObjectMsg()
        co_xwall.id = "x_axis_wall"
        co_xwall.header.frame_id = "base_link"
        co_xwall.operation = CollisionObjectMsg.ADD
        co_xwall.primitives.append(
            self._make_box([
                x_wall_thickness_m, x_wall_width_y_m, x_wall_height_m]))
        co_xwall.primitive_poses.append(
            self._make_collision_pose(
                (x_wall_x_m, 0.0, x_wall_height_m / 2.0)))

        return co_table, co_wheel, co_xwall

    def _setup_collision_objects(self, log: bool = True,
                                 blocking: bool = True) -> bool:
        if (self._moveit is None
                or not bool(
                    self.get_parameter("collision_enabled").value)):
            return False

        acquired = self._scene_collision_lock.acquire(blocking=blocking)
        if not acquired:
            return False
        try:
            co_table, co_wheel, co_xwall = self._build_collision_objects()
            with self._moveit.get_planning_scene_monitor().read_write() \
                    as scene:
                scene.apply_collision_object(co_table)
                scene.apply_collision_object(co_wheel)
                scene.apply_collision_object(co_xwall)
            if log:
                x_wall_in = float(
                    self.get_parameter("x_wall_distance_in").value)
                self.get_logger().info(
                    "Collision objects applied: "
                    f"'table_shelf' ({len(co_table.primitives)}), "
                    f"'wheelchair' ({len(co_wheel.primitives)}), "
                    f"'x_axis_wall' ({len(co_xwall.primitives)} "
                    f"@ {x_wall_in:.1f}in)")
            return True
        except Exception as e:
            self.get_logger().error(
                f"Failed to setup collision objects: {e}")
            return False
        finally:
            self._scene_collision_lock.release()

    def _refresh_collision_objects(self):
        self._setup_collision_objects(log=False, blocking=False)

    # ─────────────────────────────────────────────────────────────
    # Logging
    # ─────────────────────────────────────────────────────────────

    def _log_streams(self):
        self._append_csv(
            self.state_path,
            [self._trial_t(), self.trial_id, self.variant, self.state])
        self._append_csv(
            self.cv_path,
            [self._trial_t(), self.trial_id, self.variant,
             int(self.f_cv_current)])

    # ─────────────────────────────────────────────────────────────
    # Perception callbacks
    # ─────────────────────────────────────────────────────────────

    def _on_quality(self, msg: Float32MultiArray):
        with self._perception_lock:
            self.last_quality = list(msg.data)
        self._update_plant_ok()
        self._update_feasibility()

    def _on_plant_pose(self, msg: PoseStamped):
        """Store latest plant position from the detection node."""
        p = msg.pose.position
        if not all(math.isfinite(v) for v in (p.x, p.y, p.z)):
            return
        now = time.monotonic()
        with self._perception_lock:
            self._plant_xyz = np.array(
                [p.x, p.y, p.z], dtype=np.float64)
            self._plant_stamp = now

    def _get_quality(self) -> Optional[list]:
        with self._perception_lock:
            return (None if self.last_quality is None
                    else list(self.last_quality))

    def _update_plant_ok(self):
        q = self._get_quality()
        if not q or len(q) < Q_MIN_LEN:
            self._set_plant_ok(False)
            return
        self._set_plant_ok(
            float(q[Q_PLANT_CONF]) >= self.min_plant_conf)

    def _set_plant_ok(self, v: bool):
        now = self.get_clock().now()
        if v and not self.plant_ok_current:
            self.plant_ok_true_since = now
        if not v and self.plant_ok_current:
            self.plant_ok_true_since = None
        self.plant_ok_current = v

    def _stable_plant(self, dwell: float) -> bool:
        if not self.plant_ok_current or self.plant_ok_true_since is None:
            return False
        return ((self.get_clock().now() - self.plant_ok_true_since)
                >= Duration(seconds=dwell))

    def _update_feasibility(self):
        """Plant is feasible when: TF is ok, conf is high enough,
        valid_ratio and used_px meet thresholds."""
        q = self._get_quality()
        if not q or len(q) < Q_MIN_LEN:
            self._set_f_cv(False)
            return
        ok = (
            float(q[Q_TF_OK]) > 0.5
            and float(q[Q_PLANT_CONF]) >= self.min_plant_conf
            and float(q[Q_PLANT_VR]) >= self.min_valid_ratio
            and float(q[Q_PLANT_USED]) >= self.min_used_px
        )
        self._set_f_cv(bool(ok))

    def _set_f_cv(self, v: bool):
        now = self.get_clock().now()
        if v and not self.f_cv_current:
            self.f_cv_true_since = now
        if not v and self.f_cv_current:
            self.f_cv_true_since = None
        self.f_cv_current = v

    def _stable_feasible(self) -> bool:
        if not self.f_cv_current or self.f_cv_true_since is None:
            return False
        return ((self.get_clock().now() - self.f_cv_true_since)
                >= Duration(seconds=self.dwell_sec))

    def _target_available(self) -> bool:
        """Check if we have a recent plant pose."""
        now = time.monotonic()
        with self._perception_lock:
            if self._plant_xyz is None or self._plant_stamp is None:
                return False
            age = now - float(self._plant_stamp)
        return age < self._plant_stale_sec

    def _read_current_plant(self) -> Optional[np.ndarray]:
        """Return plant_xyz from latest perception, or None if stale."""
        now = time.monotonic()
        with self._perception_lock:
            if self._plant_xyz is None or self._plant_stamp is None:
                return None
            age = now - float(self._plant_stamp)
            if age > self._plant_stale_sec:
                return None
            return self._plant_xyz.copy()

    def _obstacle_snapshot(self) -> Tuple[float, bool, float, float]:
        now = time.monotonic()
        with self._obstacle_lock:
            closest = float(self._obstacle_closest_dist)
            blocked = bool(self._obstacle_blocked)
            min_seen = float(self._obstacle_min_depth_m)
            age = now - float(self._obstacle_last_depth_mono)
        return closest, blocked, min_seen, age

    @staticmethod
    def _fmt_depth_m(v: float) -> str:
        if math.isfinite(v):
            return f"{v:.4f}m ({v / INCH_TO_M:.2f}in)"
        return "n/a"

    def _set_obstacle_gate(self, active: bool, reason: str) -> None:
        if self._obstacle_gate_active == bool(active):
            return
        self._obstacle_gate_active = bool(active)
        closest, blocked, min_seen, age = self._obstacle_snapshot()
        self.get_logger().info(
            f"[OBS_GATE] {'ON' if active else 'OFF'} reason={reason} "
            f"blocked={int(blocked)} closest={self._fmt_depth_m(closest)} "
            f"min_seen={self._fmt_depth_m(min_seen)} depth_age={age:.2f}s")

    # ─────────────────────────────────────────────────────────────
    # Obstacle monitor
    # ─────────────────────────────────────────────────────────────

    def _on_obstacle_info(self, _msg: CameraInfo):
        pass

    def _on_obstacle_depth(self, msg: Image):
        if not self._obstacle_gate_enabled or self._obstacle_bridge is None:
            return
        try:
            depth_raw = self._obstacle_bridge.imgmsg_to_cv2(
                msg, "passthrough")
        except Exception:
            return

        d = depth_raw.astype(np.float32)
        enc = str(msg.encoding).lower()
        if enc in ("16uc1", "mono16"):
            d /= 1000.0
        if d.ndim < 2:
            return

        h, w = d.shape[:2]
        rh = int(h * self.obstacle_roi_fraction / 2.0)
        rw = int(w * self.obstacle_roi_fraction / 2.0)
        cy, cx = h // 2, w // 2
        if rh < 1 or rw < 1:
            return

        roi = d[cy - rh: cy + rh, cx - rw: cx + rw]
        valid = roi[np.isfinite(roi) & (roi > 0.01) & (roi < 5.0)]
        now = time.monotonic()

        log_msg = None
        with self._obstacle_lock:
            self._obstacle_last_depth_mono = now

            if valid.size >= self.obstacle_min_valid_pixels:
                raw_closest = float(np.percentile(
                    valid, self.obstacle_percentile))
                self._obstacle_closest_hist.append(raw_closest)
                closest = float(np.median(np.asarray(
                    self._obstacle_closest_hist, dtype=float)))
                self._obstacle_closest_dist = closest
                if math.isfinite(closest):
                    self._obstacle_min_depth_m = min(
                        float(self._obstacle_min_depth_m), closest)

                was_blocked = self._obstacle_blocked

                if closest < self.obstacle_stop_distance_m:
                    self._obstacle_stop_hits += 1
                    self._obstacle_clear_hits = 0
                elif closest > self.obstacle_resume_distance_m:
                    self._obstacle_clear_hits += 1
                    self._obstacle_stop_hits = 0
                else:
                    self._obstacle_stop_hits = 0
                    self._obstacle_clear_hits = 0

                if (not was_blocked and self._obstacle_stop_hits
                        >= self.obstacle_stop_confirm_frames):
                    self._obstacle_blocked = True
                    self._obstacle_stop_hits = 0
                elif (was_blocked and self._obstacle_clear_hits
                        >= self.obstacle_resume_confirm_frames):
                    self._obstacle_blocked = False
                    self._obstacle_clear_hits = 0

                if self._obstacle_gate_active:
                    log_msg = (
                        f"[OBS_DEPTH] valid_px={int(valid.size)} "
                        f"closest={self._fmt_depth_m(closest)} "
                        f"min_seen={self._fmt_depth_m(self._obstacle_min_depth_m)} "
                        f"blocked={int(self._obstacle_blocked)} "
                        f"hits(stop/clear)=({self._obstacle_stop_hits}/"
                        f"{self._obstacle_clear_hits})"
                    )
            else:
                self._obstacle_closest_dist = float("inf")
                if self.obstacle_fail_safe_on_no_depth:
                    self._obstacle_stop_hits += 1
                    self._obstacle_clear_hits = 0
                    if (not self._obstacle_blocked
                            and self._obstacle_stop_hits
                            >= self.obstacle_stop_confirm_frames):
                        self._obstacle_blocked = True
                        self._obstacle_stop_hits = 0
                else:
                    self._obstacle_clear_hits += 1
                    self._obstacle_stop_hits = 0
                    if (self._obstacle_blocked
                            and self._obstacle_clear_hits
                            >= self.obstacle_resume_confirm_frames):
                        self._obstacle_blocked = False
                        self._obstacle_clear_hits = 0

                if self._obstacle_gate_active:
                    log_msg = (
                        f"[OBS_DEPTH] insufficient depth valid_px={int(valid.size)} "
                        f"min_required={self.obstacle_min_valid_pixels} "
                        f"blocked={int(self._obstacle_blocked)} "
                        f"min_seen={self._fmt_depth_m(self._obstacle_min_depth_m)}"
                    )

        if log_msg is not None and self.obstacle_debug_period_s >= 0.0:
            if ((now - self._obstacle_last_debug_log_mono)
                    >= self.obstacle_debug_period_s):
                self._obstacle_last_debug_log_mono = now
                self.get_logger().info(log_msg)

    def _obstacle_is_blocked(self) -> bool:
        """Returns True if an obstacle closer than 2 inches is confirmed."""
        if not self._obstacle_gate_active:
            return False

        now = time.monotonic()
        with self._obstacle_lock:
            blocked = bool(self._obstacle_blocked)
            if (self.obstacle_fail_safe_on_no_depth
                    and self.obstacle_depth_timeout_sec > 0.0):
                age = now - self._obstacle_last_depth_mono
                if age > self.obstacle_depth_timeout_sec:
                    blocked = True
                    if not self._obstacle_timeout_active:
                        self._obstacle_timeout_active = True
                        self._obstacle_blocked = True
                else:
                    if self._obstacle_timeout_active:
                        self._obstacle_timeout_active = False
            return blocked

    # ─────────────────────────────────────────────────────────────
    # MoveIt plan/exec helpers
    # ─────────────────────────────────────────────────────────────

    def _execute_trajectory(self, trajectory) -> bool:
        res = self._moveit.execute(trajectory, controllers=[])
        if res is None:
            return True
        if isinstance(res, bool):
            return res
        success = getattr(res, "success", None)
        if isinstance(success, bool):
            return success
        error_code = getattr(res, "error_code", None)
        if error_code is not None:
            val = getattr(error_code, "val", None)
            if val is not None:
                return int(val) == MoveItErrorCodes.SUCCESS
        status = getattr(res, "status", None)
        if status is not None:
            try:
                return int(status) == GoalStatus.STATUS_SUCCEEDED
            except Exception:
                pass
        return True

    def _plan_exec_joints(self, joints: List[float],
                          planning_time: float = 2.8) -> bool:
        if self._moveit is None or self._pc is None:
            return False
        try:
            self._pc.set_start_state_to_current_state()
            gs = RobotState(self._moveit.get_robot_model())
            gs.set_joint_group_positions(self._group_name, joints)
            gs.update()
            self._pc.set_goal_state(robot_state=gs)

            params = PlanRequestParameters(self._moveit, "")
            params.planning_pipeline = self._default_planning_pipeline
            params.planning_time = float(planning_time)
            params.planning_attempts = 1
            params.max_velocity_scaling_factor = self.max_vel
            params.max_acceleration_scaling_factor = self.max_acc

            r = self._pc.plan(single_plan_parameters=params)
            if not r:
                return False
            return self._execute_trajectory(r.trajectory)
        except Exception:
            return False

    def _make_pilz_params(
            self,
            vel_scale: Optional[float] = None,
            acc_scale: Optional[float] = None) -> Optional[PlanRequestParameters]:
        if self._moveit is None:
            return None
        try:
            p = PlanRequestParameters(self._moveit, self._pilz_namespace)
            p.planning_pipeline = "pilz_industrial_motion_planner"
            p.planner_id = "LIN"
            p.planning_time = 2.0
            p.planning_attempts = 1
            p.max_velocity_scaling_factor = (
                float(PILZ_LIN_VEL) if vel_scale is None else float(vel_scale))
            p.max_acceleration_scaling_factor = (
                float(PILZ_LIN_ACC) if acc_scale is None else float(acc_scale))
            return p
        except Exception:
            return None

    def _plan_exec_pose(self, pose: PoseStamped,
                        linear: bool,
                        allow_fallback: bool = True,
                        linear_vel_scale: Optional[float] = None,
                        linear_acc_scale: Optional[float] = None) -> bool:
        if self._moveit is None or self._pc is None:
            return False
        try:
            ee = str(self.get_parameter("ee_link").value)
            self._pc.set_start_state_to_current_state()
            self._pc.set_goal_state(
                pose_stamped_msg=pose, pose_link=ee)

            if linear and self._pilz_params is None and not allow_fallback:
                return False

            if linear and self._pilz_params is not None:
                pilz_params = self._pilz_params
                if (linear_vel_scale is not None
                        or linear_acc_scale is not None):
                    pilz_params = self._make_pilz_params(
                        vel_scale=linear_vel_scale,
                        acc_scale=linear_acc_scale)
                if pilz_params is not None:
                    r = self._pc.plan(
                        single_plan_parameters=pilz_params)
                else:
                    r = None
                if r:
                    return self._execute_trajectory(r.trajectory)
                if not allow_fallback:
                    return False

            params = PlanRequestParameters(self._moveit, "")
            params.planning_pipeline = self._default_planning_pipeline
            params.planning_time = 2.0
            params.planning_attempts = 1
            params.max_velocity_scaling_factor = self.max_vel
            params.max_acceleration_scaling_factor = self.max_acc

            r = self._pc.plan(single_plan_parameters=params)
            if not r:
                return False
            return self._execute_trajectory(r.trajectory)
        except Exception:
            return False

    def _plan_exec_named(self, name: str) -> bool:
        if self._moveit is None or self._pc is None:
            return False
        try:
            self._pc.set_start_state_to_current_state()
            self._pc.set_goal_state(configuration_name=name)

            params = PlanRequestParameters(self._moveit, "")
            params.planning_pipeline = self._default_planning_pipeline
            params.planning_time = 2.0
            params.planning_attempts = 1
            params.max_velocity_scaling_factor = self.max_vel
            params.max_acceleration_scaling_factor = self.max_acc

            r = self._pc.plan(single_plan_parameters=params)
            return self._execute_trajectory(r.trajectory) if r else False
        except Exception:
            return False

    def _wait_future(self, future, timeout_s: float) -> bool:
        t0 = time.monotonic()
        while rclpy.ok() and not future.done():
            if (time.monotonic() - t0) > timeout_s:
                return False
            time.sleep(0.01)
        return future.done()

    def _enable_servo_if_needed(self) -> bool:
        if not self.straight_vector_enable_servo:
            return True
        if self._servo_enabled:
            return True

        if not self._servo_start_cli.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn(
                f"Servo start service unavailable: {self.servo_start_service}")
            return False

        future = self._servo_start_cli.call_async(Trigger.Request())
        if not self._wait_future(future, timeout_s=1.5):
            self.get_logger().warn("Servo start timeout")
            return False

        try:
            result = future.result()
        except Exception as e:
            self.get_logger().warn(f"Servo start failed: {e}")
            return False

        ok = bool(getattr(result, "success", False))
        if not ok:
            self.get_logger().warn(
                f"Servo start rejected: {getattr(result, 'message', '')}")
            return False

        self._servo_enabled = True
        self.get_logger().info("Servo enabled for straight vector move")
        return True

    def _publish_twist(self, vx: float, vy: float, vz: float) -> None:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.straight_vector_frame
        msg.twist.linear.x = float(vx)
        msg.twist.linear.y = float(vy)
        msg.twist.linear.z = float(vz)
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0
        self._straight_twist_pub.publish(msg)

    def _move_straight_vector_servo(
            self,
            direction: np.ndarray,
            distance_m: float,
            label: str,
            obstacle_check_cb: Optional[Callable[[str], bool]] = None) -> bool:
        if distance_m <= 0.0:
            return True

        n = float(np.linalg.norm(direction))
        if n < 1e-6:
            self.get_logger().warn(
                f"{label}: zero direction for straight vector")
            return False
        unit = direction / n

        speed = max(0.002, float(self.straight_vector_speed_mps))
        hz = max(5.0, float(self.straight_vector_hz))
        dt = 1.0 / hz
        duration = float(distance_m) / speed
        vx, vy, vz = (unit * speed).tolist()

        if not self._enable_servo_if_needed():
            return False

        if (self.straight_vector_require_subscriber
                and self._straight_twist_pub.get_subscription_count() == 0):
            self.get_logger().warn(
                f"{label}: no subscribers on {self.straight_vector_topic}")
            return False

        self.get_logger().info(
            f"{label}: Cartesian vector move "
            f"dist={distance_m:.3f}m speed={speed:.3f}m/s "
            f"frame={self.straight_vector_frame} "
            f"v=[{vx:.4f},{vy:.4f},{vz:.4f}]")

        t0 = time.monotonic()
        while rclpy.ok():
            elapsed = time.monotonic() - t0
            if elapsed >= duration:
                break
            if obstacle_check_cb is not None:
                if not obstacle_check_cb(f"{label} (forward)"):
                    for _ in range(4):
                        self._publish_twist(0.0, 0.0, 0.0)
                        time.sleep(0.02)
                    return False
            self._publish_twist(vx, vy, vz)
            time.sleep(dt)

        # Send zero twist a few times to guarantee stop.
        for _ in range(4):
            self._publish_twist(0.0, 0.0, 0.0)
            time.sleep(0.02)
        return True

    def _move_straight_vector_lin_steps(
            self,
            start_tip: np.ndarray,
            tool_fwd_axis: np.ndarray,
            quat: Tuple[float, float, float, float],
            direction: np.ndarray,
            distance_m: float,
            tip_offset_m: float,
            label: str,
            obstacle_check_cb: Optional[Callable[[str], bool]] = None) -> bool:
        if distance_m <= 0.0:
            return True

        dn = float(np.linalg.norm(direction))
        fn = float(np.linalg.norm(tool_fwd_axis))
        if dn < 1e-6 or fn < 1e-6:
            self.get_logger().warn(f"{label}: invalid lin_step vectors")
            return False

        d_unit = direction / dn
        fwd_unit = tool_fwd_axis / fn
        step_m = float(self.straight_vector_step_m)
        step_m = max(0.10 * INCH_TO_M, step_m)
        steps = max(1, int(math.ceil(float(distance_m) / step_m)))

        self.get_logger().info(
            f"{label}: lin_step fallback "
            f"dist={distance_m:.3f}m step={step_m:.3f}m n={steps}")

        for idx in range(1, steps + 1):
            if obstacle_check_cb is not None:
                if not obstacle_check_cb(f"{label} (step {idx}/{steps})"):
                    return False
            s = min(float(distance_m), idx * step_m)
            tip_target = start_tip + d_unit * s
            ee_target = tip_target - fwd_unit * float(tip_offset_m)
            pose = self._make_pose(ee_target, quat)
            ok = self._exec_forward_step_with_ompl_fallback(
                pose=pose,
                step_idx=idx,
                step_total=steps,
                step_label=label)
            if not ok:
                self.get_logger().warn(
                    f"{label}: lin_step failed at step {idx}/{steps}")
                return False
        return True

    def _exec_forward_step_with_ompl_fallback(
            self,
            pose: PoseStamped,
            step_idx: int,
            step_total: int,
            step_label: str) -> bool:
        # Forward-stage policy:
        # 1) Try pure Pilz LIN first (no OMPL fallback).
        # 2) If LIN fails, run one OMPL move with obstacle gate disabled.
        if self._plan_exec_pose(
                pose, True, allow_fallback=False,
                linear_vel_scale=self.sweep_pilz_vel,
                linear_acc_scale=self.sweep_pilz_acc):
            return True

        self.get_logger().warn(
            f"{step_label}: LIN failed at step {step_idx}/{step_total}; "
            "using OMPL fallback with obstacle gate OFF")

        gate_was_on = bool(self._obstacle_gate_active)
        if gate_was_on:
            self._set_obstacle_gate(
                False, f"ompl_fallback_{step_label}_{step_idx}")
        ok = self._plan_exec_pose(
            pose, False, allow_fallback=True)
        if gate_was_on:
            self._set_obstacle_gate(
                True, f"resume_lin_{step_label}_{step_idx}")
        return bool(ok)

    def _move_straight_vector(
            self,
            direction: np.ndarray,
            distance_m: float,
            label: str,
            start_tip: Optional[np.ndarray] = None,
            tool_fwd_axis: Optional[np.ndarray] = None,
            quat: Optional[Tuple[float, float, float, float]] = None,
            tip_offset_m: Optional[float] = None,
            obstacle_check_cb: Optional[Callable[[str], bool]] = None) -> bool:
        backend = str(self.straight_vector_backend).strip().lower()

        if backend in ("auto", "servo"):
            if self._move_straight_vector_servo(
                    direction, distance_m, label,
                    obstacle_check_cb=obstacle_check_cb):
                return True
            if backend == "servo":
                return False
            self.get_logger().warn(
                f"{label}: servo vector unavailable, using lin_step fallback")

        if backend in ("auto", "lin_step"):
            if (start_tip is None or tool_fwd_axis is None
                    or quat is None or tip_offset_m is None):
                self.get_logger().warn(
                    f"{label}: missing inputs for lin_step fallback")
                return False
            return self._move_straight_vector_lin_steps(
                start_tip=start_tip,
                tool_fwd_axis=tool_fwd_axis,
                quat=quat,
                direction=direction,
                distance_m=distance_m,
                tip_offset_m=float(tip_offset_m),
                label=label,
                obstacle_check_cb=obstacle_check_cb)

        self.get_logger().warn(
            f"{label}: unsupported straight_vector_backend={backend}")
        return False

    def _command_gripper(self, position: float, label: str) -> bool:
        timeout_s = max(0.1, float(self.gripper_command_timeout_s))
        if not self._gripper_client.wait_for_server(timeout_sec=timeout_s):
            self.get_logger().warn(
                f"Gripper action unavailable for '{label}' "
                f"(topic={self.gripper_action_name})")
            return False

        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = float(self.gripper_max_effort)

        send_future = self._gripper_client.send_goal_async(goal)
        t0 = time.monotonic()
        while rclpy.ok() and not send_future.done():
            if (time.monotonic() - t0) > timeout_s:
                self.get_logger().warn(
                    f"Gripper send timeout for '{label}'")
                return False
            time.sleep(0.01)

        if not send_future.done():
            return False

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn(
                f"Gripper goal rejected for '{label}'")
            return False

        result_future = goal_handle.get_result_async()
        t1 = time.monotonic()
        while rclpy.ok() and not result_future.done():
            if (time.monotonic() - t1) > timeout_s:
                self.get_logger().warn(
                    f"Gripper result timeout for '{label}'")
                return False
            time.sleep(0.01)

        if not result_future.done():
            return False

        result = result_future.result()
        status = getattr(result, "status", None)
        if status is not None and int(status) != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().warn(
                f"Gripper failed for '{label}' (status={int(status)})")
            return False

        self.get_logger().info(
            f"Gripper {label}: pos={float(position):.4f} "
            f"effort={self.gripper_max_effort:.1f}")
        return True

    def _make_pose(self, pos: np.ndarray,
                   quat: Tuple[float, float, float, float]) -> PoseStamped:
        m = PoseStamped()
        m.header.frame_id = str(self.get_parameter("base_frame").value)
        m.header.stamp = self.get_clock().now().to_msg()
        m.pose.position.x = float(pos[0])
        m.pose.position.y = float(pos[1])
        m.pose.position.z = float(pos[2])
        m.pose.orientation.x = quat[0]
        m.pose.orientation.y = quat[1]
        m.pose.orientation.z = quat[2]
        m.pose.orientation.w = quat[3]
        return m

    # ─────────────────────────────────────────────────────────────
    # State transitions
    # ─────────────────────────────────────────────────────────────

    def _enter_hold(self):
        self.state = HOLD

    def _enter_assist(self, start_scan_idx: int = 0):
        self.state = ASSIST
        self.scan_idx = int(start_scan_idx) % len(SCAN_POSES)
        self.scan_reached_mono = None

    def _mark_started(self):
        if self.started_flag == 0:
            self.started_flag = 1
            self.t_start_s = self._trial_t()
            self.get_logger().info(
                f"[START] t={self.t_start_s:.3f}s "
                f"f_cv={int(self.f_cv_current)}")

    def _start_execute(self) -> bool:
        now_mono = time.monotonic()
        if self._attempt_idx >= self.max_execute_attempts:
            if not self._max_attempts_exceeded_logged:
                self.get_logger().error(
                    f"Max execute attempts reached "
                    f"({self.max_execute_attempts})")
                self._max_attempts_exceeded_logged = True
            return False
        if now_mono < self._next_retry_mono:
            return False

        self._max_attempts_exceeded_logged = False
        self.state = EXECUTE
        self._mark_started()
        self._attempt_idx += 1
        self._attempt_active = True
        self._attempt_start_s = self._trial_t()
        self._t_start_mono = time.monotonic()
        self._obstacle_min_depth_m = float("inf")
        self._obstacle_last_debug_log_mono = 0.0

        p = self._read_current_plant()
        if p is not None:
            self.get_logger().info(
                f"[TASK] Starting attempt {self._attempt_idx} "
                f"from plant=[{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}]")
        else:
            self.get_logger().info(
                f"[TASK] Starting attempt {self._attempt_idx} "
                "(plant pose unavailable/stale)")

        self._sweep_result = None
        self._sweep_thread = threading.Thread(
            target=self._run_sweep_sequence, daemon=True)
        self._sweep_thread.start()
        return True

    def _record_attempt_end(self, result: str, detail: str = ""):
        if not self._attempt_active:
            return
        t_end = self._trial_t()
        outcome = str(result).strip()
        d = str(detail).strip()
        if d:
            outcome = f"{outcome}:{d}"
        if len(outcome) > 240:
            outcome = outcome[:240]
        self._append_csv(
            self.attempts_path,
            [self.trial_id, self.variant,
             int(self.started_flag),
             self._attempt_start_s, t_end, outcome])
        self._attempt_active = False
        self._attempt_start_s = float("nan")

    def _finish(self, outcome: str):
        if self._final_saved:
            return
        self._final_saved = True
        self.outcome = outcome
        if outcome == "success":
            self.t_success_s = self._trial_t()
        else:
            self.t_success_s = float("nan")

        try:
            self._write_metrics_files()
        except Exception as e:
            self.get_logger().error(f"Metrics write failed: {e}")

        self.get_logger().info(
            f"[FINISH] outcome={self.outcome} "
            f"started={self.started_flag} "
            f"log_dir={self.log_dir} "
            f"global_summary={self.global_summary_csv}")

    # ─────────────────────────────────────────────────────────────
    # Metrics
    # ─────────────────────────────────────────────────────────────

    def _compute_trial_metrics(self) -> Dict[str, Any]:
        started = int(self.started_flag)

        time_to_success = float("nan")
        if (self.outcome == "success" and started == 1
                and math.isfinite(self.t_success_s)
                and math.isfinite(self.t_start_s)):
            time_to_success = float(self.t_success_s - self.t_start_s)

        false_start = 0
        gt_action_at_start = -1
        if (started == 1 and self.model is not None
                and math.isfinite(self.t_start_s)):
            i = _nearest_index(
                self.model["center_time_s"], self.t_start_s)
            gt_action_at_start = int(self.model["gt_action"][i])
            false_start = 1 if gt_action_at_start == 0 else 0

        cv_infeas_start = 0
        f_cv_at_start = (
            int(self.f_cv_current) if started == 1 else -1)
        if started == 1 and f_cv_at_start == 0:
            cv_infeas_start = 1

        flaps = 0
        if self.model is not None:
            cs = [str(x).strip().upper()
                  for x in self.model["commit_state"]]
            cs = [x for x in cs if x in ("HOLD", "COMMIT")]
            flaps = _count_flaps(cs)

        return {
            "started": started,
            "time_to_success": time_to_success,
            "false_start": int(false_start),
            "gt_action_at_start": int(gt_action_at_start),
            "cv_infeas_start": int(cv_infeas_start),
            "f_cv_at_start": int(f_cv_at_start),
            "flaps": int(flaps),
        }

    def _write_metrics_files(self):
        m = self._compute_trial_metrics()

        self._append_csv(
            self.metrics_trial_path,
            [
                self.trial_id, self.variant,
                m["started"], self.outcome,
                self.t_start_s, self.t_success_s,
                m["time_to_success"],
                m["false_start"], m["gt_action_at_start"],
                m["cv_infeas_start"], m["f_cv_at_start"],
                m["flaps"]
            ],
        )

        self._append_csv(
            self.global_summary_csv,
            [
                self.trial_id,
                self.variant,
                self.log_dir,
                self.pred_csv,
                m["started"],
                self.outcome,
                self.t_start_s,
                self.t_success_s,
                m["false_start"],
                m["cv_infeas_start"],
                m["flaps"],
                m["time_to_success"],
            ],
        )

    # ─────────────────────────────────────────────────────────────
    # Commit service
    # ─────────────────────────────────────────────────────────────

    def _on_commit(self, req, res):
        if self.state != HOLD:
            res.success = False
            res.message = f"Cannot commit in state {self.state}"
            return res
        if self.variant == "feas_only":
            res.success = False
            res.message = "feas_only auto-starts"
            return res
        res.message = self._do_commit()
        res.success = res.message.startswith("Started")
        return res

    def _do_commit(self) -> str:
        def _start_or_reason() -> str:
            if self._start_execute():
                return "Started execute"
            if self._attempt_idx >= self.max_execute_attempts:
                return (
                    f"Blocked: max execute attempts reached "
                    f"({self.max_execute_attempts})")
            if time.monotonic() < self._next_retry_mono:
                wait_s = max(0.0, self._next_retry_mono - time.monotonic())
                return f"Blocked: retry cooldown ({wait_s:.1f}s)"
            return "Blocked: could not start execute"

        if self.variant in ("commit_only", "hold_commit"):
            if not self._target_available():
                return "Blocked: no target"
            return _start_or_reason()

        if self.variant == "hac":
            if self._target_available() and self._stable_feasible():
                return _start_or_reason()
            self._enter_assist(
                start_scan_idx=self._plant_found_scan_idx or 0)
            return "Entered ASSIST"

        return "Unknown"

    # ─────────────────────────────────────────────────────────────
    # Scan
    # ─────────────────────────────────────────────────────────────

    def _move_to_scan_pose(self) -> bool:
        p = list(SCAN_POSES[self.scan_idx])
        ok = self._plan_exec_joints(p)
        if ok:
            self.scan_reached_mono = time.monotonic()
            return True
        self.scan_reached_mono = None
        return False

    # ─────────────────────────────────────────────────────────────
    # Sweep sequence (runs in thread)
    # ─────────────────────────────────────────────────────────────

    def _run_sweep_sequence(self):
        """Execute the plant sweep motion — simple L-shaped push.

        1. Locate plant (already done).
        2. Compute fwd (base->plant XY direction) and right (perpendicular).
        3. Waypoints (all in base_link, at plant Z height):
             start  = plant - fwd*15in - right*5in  (15in behind, 5in left)
             after_straight = start + fwd*15in       (now level with plant, 5in left)
             after_push     = after_straight + right*5in  (pushes plant right)
        4. Motion:
             a) Joint move to start + hover_above
             b) LIN lower to start
             c) LIN go straight 15in (+fwd)
             d) ** obstacle gate active **
             e) LIN go right 5in (+right) -- pushes plant
             f) ** obstacle gate off **
             g) LIN raise up
             h) Joint move home
        """
        try:
            plant_pos = self._read_current_plant()
            if plant_pos is None:
                self._sweep_result = (False, "No plant position")
                return
            self.get_logger().info(
                "[TASK] Plant recognized, executing move sequence")

            # Read parameters
            left_m = float(self.get_parameter("left_offset_in").value) * INCH_TO_M
            straight_m = float(self.get_parameter("straight_distance_in").value) * INCH_TO_M
            push_m = float(self.get_parameter("sweep_right_in").value) * INCH_TO_M
            hover_m = float(self.get_parameter("hover_in").value) * INCH_TO_M
            back_m = float(self.get_parameter("approach_back_in").value) * INCH_TO_M
            tip_m = float(self.get_parameter("tip_offset_in").value) * INCH_TO_M

            up = np.array([0.0, 0.0, 1.0])

            # fwd = direction from robot base to plant (XY plane)
            plant_xy = np.array([plant_pos[0], plant_pos[1], 0.0])
            d = np.linalg.norm(plant_xy)
            if d < 0.01:
                fwd = np.array([1.0, 0.0, 0.0])
            else:
                fwd = plant_xy / d

            # right = perpendicular to fwd in XY plane
            right = np.cross(fwd, up)
            rn = np.linalg.norm(right)
            if rn < 1e-6:
                right = np.array([1.0, 0.0, 0.0])
            else:
                right = right / rn

            self.get_logger().info(
                f"Plant at [{plant_pos[0]:.3f}, {plant_pos[1]:.3f}, {plant_pos[2]:.3f}] "
                f"fwd=[{fwd[0]:.3f}, {fwd[1]:.3f}] "
                f"right=[{right[0]:.3f}, {right[1]:.3f}]")

            # Tool orientation: EE Z-axis -> fwd, EE Y-axis -> up
            quat = _make_orientation_quat(
                fwd, up,
                tool_forward_axis=str(self.get_parameter("tool_forward_axis").value),
                tool_up_axis=str(self.get_parameter("tool_up_axis").value),
            )

            # EE position = tip - fwd * tip_offset
            def ee(tip):
                return tip - fwd * tip_m

            # -- Waypoints (tip positions) --
            # Start: 15in behind plant, 5in to the left
            start = plant_pos.copy()
            start -= fwd * straight_m    # 15in behind
            start -= right * left_m      # 5in left

            # After going straight 15in toward plant
            after_straight = start + fwd * straight_m

            # After pushing right 5in
            after_push = after_straight + right * push_m

            self.get_logger().info(
                f"Waypoints: start=[{start[0]:.3f},{start[1]:.3f},{start[2]:.3f}] "
                f"after_straight=[{after_straight[0]:.3f},{after_straight[1]:.3f},{after_straight[2]:.3f}] "
                f"after_push=[{after_push[0]:.3f},{after_push[1]:.3f},{after_push[2]:.3f}]")

            # -- Helper: check obstacles per variant --
            hold_pause_latched = False

            def check_obstacle(label):
                nonlocal hold_pause_latched
                if not self._obstacle_gate_active:
                    return True
                blocked = self._obstacle_is_blocked()
                closest, _, min_seen, age = self._obstacle_snapshot()
                self.get_logger().info(
                    f"[OBS_CHECK] at={label} blocked={int(blocked)} "
                    f"closest={self._fmt_depth_m(closest)} "
                    f"min_seen={self._fmt_depth_m(min_seen)} "
                    f"depth_age={age:.2f}s")
                if self.variant == "feas_only":
                    if blocked:
                        self._sweep_result = (
                            False, f"feas_only abort: obstacle at {label}")
                        return False
                    return True
                if self.variant == "hac":
                    if blocked:
                        self.get_logger().warn(
                            f"hac: obstacle at {label}, waiting...")
                        while rclpy.ok() and self._obstacle_is_blocked():
                            time.sleep(0.05)
                        self.get_logger().info(f"hac: cleared at {label}")
                    return True
                if self.variant == "hold_commit":
                    if blocked:
                        if not hold_pause_latched:
                            self.get_logger().warn(
                                f"hold_commit: obstacle at {label}, pausing "
                                f"{self.hold_commit_obstacle_pause_sec:.1f}s")
                            time.sleep(self.hold_commit_obstacle_pause_sec)
                            hold_pause_latched = True
                    else:
                        hold_pause_latched = False
                return True

            # -- Execute motion --

            # Step 1: Joint move to start + hover (safe approach)
            approach_pos = start - fwd * back_m + up * hover_m
            self.get_logger().info("Step 1: approach (joint)")
            if not self._plan_exec_pose(self._make_pose(ee(approach_pos), quat), False):
                self._sweep_result = (False, "Plan failed: approach")
                return
            self.get_logger().info("Step 1 done")

            # Step 2: LIN hover above start
            self.get_logger().info("Step 2: hover above start (LIN)")
            if not self._plan_exec_pose(self._make_pose(ee(start + up * hover_m), quat), True):
                self._sweep_result = (False, "Plan failed: hover")
                return
            self.get_logger().info("Step 2 done")

            # Step 3: LIN lower to start height
            self.get_logger().info("Step 3: lower to start (LIN)")
            if not self._plan_exec_pose(self._make_pose(ee(start.copy()), quat), True):
                self._sweep_result = (False, "Plan failed: lower")
                return
            self.get_logger().info("Step 3 done")

            # Step 4: go straight 15in toward plant with obstacle handling ON.
            if self._obstacle_gate_enabled:
                self._set_obstacle_gate(True, "forward_stage_start")

            if self.straight_use_cartesian_vector:
                self.get_logger().info("Step 4: go straight 15in (Cartesian vector)")
                if not self._move_straight_vector(
                        direction=fwd.copy(),
                        distance_m=straight_m,
                        label="straight_15in",
                        start_tip=start.copy(),
                        tool_fwd_axis=fwd.copy(),
                        quat=quat,
                        tip_offset_m=tip_m,
                        obstacle_check_cb=check_obstacle):
                    self._set_obstacle_gate(False, "forward_stage_failed")
                    if self._sweep_result is None:
                        self._sweep_result = (
                            False, "Cartesian vector failed: straight")
                    return
            else:
                self.get_logger().info("Step 4: go straight 15in (LIN segmented)")
                step_m = max(0.10 * INCH_TO_M, self.straight_vector_step_m)
                n_steps = max(1, int(math.ceil(float(straight_m) / step_m)))
                for idx in range(1, n_steps + 1):
                    if not check_obstacle(f"forward LIN [{idx}/{n_steps}]"):
                        self._set_obstacle_gate(
                            False, "forward_lin_blocked")
                        if self._sweep_result is None:
                            self._sweep_result = (
                                False,
                                f"Obstacle blocked during forward LIN "
                                f"[{idx}/{n_steps}]")
                        return
                    s = min(float(straight_m), idx * step_m)
                    tip_i = start + fwd * s
                    if not self._exec_forward_step_with_ompl_fallback(
                            pose=self._make_pose(ee(tip_i), quat),
                            step_idx=idx,
                            step_total=n_steps,
                            step_label="forward_lin"):
                        self._set_obstacle_gate(
                            False, "forward_lin_plan_failed")
                        self._sweep_result = (
                            False,
                            f"Plan failed: straight [{idx}/{n_steps}]")
                        return

            # Aligned left of plant: obstacle handling OFF from here onward.
            if self._obstacle_gate_active:
                self._set_obstacle_gate(
                    False, "left_aligned_before_push")
            self.get_logger().info("Step 4 done (left aligned)")

            self.get_logger().info("Step 5: push right 5in (LIN)")
            if not self._plan_exec_pose(self._make_pose(ee(after_push.copy()), quat), True):
                self._sweep_result = (False, "Plan failed: push right")
                return
            self.get_logger().info("Step 5 done")

            # Step 6: LIN raise up
            self.get_logger().info("Step 6: raise (LIN)")
            if not self._plan_exec_pose(
                    self._make_pose(ee(after_push + up * hover_m), quat), True):
                self._sweep_result = (False, "Plan failed: raise")
                return
            self.get_logger().info("Step 6 done")

            # Step 7: Go home
            self.get_logger().info("Step 7: go home")
            if not self._plan_exec_named(
                    str(self.get_parameter("home_named_target").value)):
                self._sweep_result = (False, "Plan failed: home")
                return
            self.get_logger().info("Step 7 done")

            self._sweep_result = (True, "Done")

        except Exception as e:
            self._set_obstacle_gate(False, "exception")
            self._sweep_result = (False, f"Exception: {e}")
    # ─────────────────────────────────────────────────────────────
    # Main tick
    # ─────────────────────────────────────────────────────────────

    def _tick(self):
        if self._finished:
            return

        self._log_streams()

        # HAC: if obstacle during PRE_SCAN or HOLD, go to ASSIST
        if (self.variant == "hac"
                and self.state in (PRE_SCAN, HOLD)
                and self._obstacle_gate_active
                and self._obstacle_is_blocked()):
            self._enter_assist(
                start_scan_idx=self._plant_found_scan_idx or 0)
            return

        # ─── PRE_SCAN ──────────────────────────────────────────
        if self.state == PRE_SCAN:
            if self._stable_plant(self.prescan_dwell):
                self._plant_found_scan_idx = self.scan_idx
                self._enter_hold()
                return

            if self.prescan_cycles > self.prescan_max_cycles:
                self._enter_hold()
                return

            if self.scan_reached_mono is None:
                self._move_to_scan_pose()
                return

            elapsed = time.monotonic() - float(self.scan_reached_mono)
            if elapsed < (self.prescan_settle + self.scan_extra_pause):
                return

            self.scan_idx = (self.scan_idx + 1) % len(SCAN_POSES)
            if self.scan_idx == 0:
                self.prescan_cycles += 1
            self.scan_reached_mono = None
            return

        # ─── ASSIST ─────────────────────────────────────────────
        if self.state == ASSIST:
            if self._stable_plant(self.prescan_dwell):
                self._plant_found_scan_idx = self.scan_idx
                self._enter_hold()
                return

            if self.scan_reached_mono is None:
                self._move_to_scan_pose()
                return

            elapsed = time.monotonic() - float(self.scan_reached_mono)
            if elapsed < (self.prescan_settle + self.scan_extra_pause):
                return

            self.scan_idx = (self.scan_idx + 1) % len(SCAN_POSES)
            self.scan_reached_mono = None
            return

        # ─── HOLD ───────────────────────────────────────────────
        if self.state == HOLD:
            # feas_only: start as soon as plant visible + feasible
            if self.variant == "feas_only":
                if (self._stable_feasible()
                        and self._target_available()):
                    self._start_execute()
                return

            # commit_only: wait for commit time, then go
            if self.variant == "commit_only":
                if self.t_commit_s is None:
                    if self._target_available():
                        self.get_logger().warn(
                            "commit_only: no commit with gt==1, "
                            "starting on target available")
                        self._start_execute()
                    return
                if self._trial_t() >= float(self.t_commit_s):
                    if self._target_available():
                        self._start_execute()
                return

            # hold_commit: wait for commit time AND feasibility
            if self.variant == "hold_commit":
                if self.t_commit_s is None:
                    if (self._target_available()
                            and self._stable_feasible()):
                        self.get_logger().warn(
                            "hold_commit: no commit with gt==1, "
                            "starting on feasibility + target")
                        self._start_execute()
                    return
                if self._trial_t() >= float(self.t_commit_s):
                    if (self._target_available()
                            and self._stable_feasible()):
                        self._start_execute()
                return

            # hac: wait for commit, then require feasibility
            if self.variant == "hac":
                if (self.t_commit_s is not None
                        and self._trial_t()
                        < float(self.t_commit_s)):
                    return  # still waiting for commit time
                if (self._target_available()
                        and self._stable_feasible()):
                    self._start_execute()
                else:
                    if (self.t_commit_s is not None
                            and self._trial_t()
                            >= float(self.t_commit_s)):
                        self._enter_assist(
                            start_scan_idx=(
                                self._plant_found_scan_idx or 0))
                return

        # ─── EXECUTE ────────────────────────────────────────────
        if self.state == EXECUTE:
            if self._sweep_thread is None:
                return
            if self._sweep_thread.is_alive():
                return

            self._sweep_thread = None
            success, msg = self._sweep_result or (False, "No result")
            self._record_attempt_end(
                "success" if success else "failed", msg)

            if success:
                self.get_logger().info(f"EXECUTE success: {msg}")
                self._finish("success")

                if self.stop_after_success:
                    self._request_shutdown()
                    return
                self._enter_hold()
                return

            self.get_logger().warn(f"EXECUTE failed: {msg}")
            self._next_retry_mono = (
                time.monotonic() + float(self.retry_cooldown_s))

            if self._attempt_idx >= self.max_execute_attempts:
                self.get_logger().error(
                    f"Stopping retries after {self._attempt_idx} failed "
                    f"attempt(s); max_execute_attempts="
                    f"{self.max_execute_attempts}")
                self._finish("failed")
                if self.stop_after_failure:
                    self._request_shutdown()
                    return

            if (self.variant == "hac"
                    and ("rescan" in str(msg).lower()
                         or "perception" in str(msg).lower())):
                self._enter_assist(
                    start_scan_idx=(
                        self._plant_found_scan_idx or 0))
                return

            self._enter_hold()

    def _watchdog_timeout(self):
        if self._finished:
            return
        if (self.max_runtime_s > 0.0
                and self._trial_t() > self.max_runtime_s):
            self.get_logger().error("Timeout: shutting down")
            self._record_attempt_end("timeout", "watchdog")
            self._finish("timeout")
            self._request_shutdown()

    def _handle_keyboard_interrupt(self):
        if self._attempt_active:
            self._record_attempt_end(
                "interrupted", "keyboard_interrupt")
        self._finish("interrupted")
        self._finished = True


# =====================================================================
# Entry point
# =====================================================================

def main():
    rclpy.init()
    node = PlantMoveSupervisor()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        try:
            node._handle_keyboard_interrupt()
        except Exception:
            pass
    finally:
        try:
            node._cleanup_motion_resources()
        except Exception:
            pass
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
