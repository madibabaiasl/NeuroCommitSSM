#!/usr/bin/env python3
"""
clock_pick_supervisor_metrics.py

Simplified clock pick task supervisor.  Subscribes to the clock pose published
by clock_pose_node_metrics.py, then executes:

    WAIT_POSE → OPEN_GRIPPER → SCAN → [detection] → LIN_APPROACH → MOVE_GRASP → GRASP
    → RELEASE → RETREAT → DONE

There is NO lift or place sequence — the robot grasps the clock corner, then
releases and retreats.

Variants (obstacle-aware motion):
    commit_only  — no retry on obstacle, returns False
    feas_only    — raises abort, sets task result to failure
    hold_commit  — pauses N seconds then retries
    hac          — waits until obstacle clears then retries

Run:
  ros2 run bottle_grasping clock_pick_supervisor_metrics --ros-args \\
    -p variant:=commit_only \\
    -p trial_id:=clock_pick_trial \\
    -p log_dir:=/tmp/clock_pick_logs \\
    -p global_summary_csv:=/tmp/clock_pick_metrics_summary.csv
"""

import math
import os
import threading
import time
import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R
import yaml

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.time import Time

from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Pose
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker
from action_msgs.msg import GoalStatus

from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive
from moveit_configs_utils import MoveItConfigsBuilder
from moveit.planning import MoveItPy, PlanRequestParameters
from moveit.core.robot_state import RobotState

from control_msgs.action import GripperCommand

from pymoveit2.robots import kinova
from ament_index_python.packages import get_package_share_directory
from rclpy.duration import Duration

# ============================
# Constants
# ============================
INCH_TO_M = 0.0254
MOVEIT_PKG = "kinova_gen3_6dof_robotiq_2f_85_moveit_config"
ARM_JOINTS = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
]
CLOCK_LENGTH_M = 6.0 * INCH_TO_M    # 0.1524 m
CLOCK_WIDTH_M  = 1.5 * INCH_TO_M    # 0.0381 m
CLOCK_HEIGHT_M = 3.0 * INCH_TO_M    # 0.0762 m
GRASP_END_INSET_M = 0.5 * INCH_TO_M # 0.0127 m

SUPPORTED_VARIANTS = {"commit_only", "feas_only", "hold_commit", "hac"}

@dataclass
class MotionDiag:
    phase: str
    ok: bool
    cartesian: bool
    planning_time_s: float
    attempts: int
    elapsed_s: float
    detail: str


class _ObstacleAbort(Exception):
    pass


# ============================
# Gripper helper
# ============================

class Gripper:
    def __init__(self, node: Node):
        self.node = node
        self._client = ActionClient(
            node,
            GripperCommand,
            "/robotiq_gripper_controller/gripper_cmd",
            callback_group=ReentrantCallbackGroup(),
        )
        self._goal_handle = None
        self._result_future = None

    def wait_for_server(self, timeout_s: float = 5.0) -> bool:
        return self._client.wait_for_server(timeout_sec=timeout_s)

    def command(self, position: float, max_effort: float = 100.0, timeout_s: float = 5.0):
        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = float(max_effort)
        self._result_future = None
        self._goal_handle = None
        send_future = self._client.send_goal_async(goal)
        t0 = time.time()
        while rclpy.ok() and not send_future.done():
            if (time.time() - t0) > timeout_s:
                self.node.get_logger().error("Gripper send_goal timeout")
                return False
            time.sleep(0.01)

        self._goal_handle = send_future.result()
        if self._goal_handle is None or not self._goal_handle.accepted:
            self.node.get_logger().error("Gripper goal rejected")
            return False
        self._result_future = self._goal_handle.get_result_async()
        return True

    def wait(self, timeout_s: float = 5.0) -> bool:
        if self._result_future is None:
            return False
        t0 = time.time()
        while rclpy.ok() and not self._result_future.done():
            if (time.time() - t0) > timeout_s:
                self.node.get_logger().error("Gripper result timeout")
                return False
            time.sleep(0.01)
        try:
            res = self._result_future.result()
            status = getattr(res, "status", GoalStatus.STATUS_UNKNOWN)
            return int(status) == GoalStatus.STATUS_SUCCEEDED
        except Exception:
            return False

    def cancel(self):
        if self._goal_handle is not None:
            cancel_future = self._goal_handle.cancel_goal_async()
            t0 = time.time()
            while rclpy.ok() and not cancel_future.done():
                if (time.time() - t0) > 2.0:
                    break
                time.sleep(0.01)


# ============================
# Utility functions
# ============================

def infer_srdf() -> Tuple[str, str, str]:
    try:
        share = Path(get_package_share_directory(MOVEIT_PKG))
        srdf = next(share.joinpath("config").glob("*.srdf"))
        root = ET.parse(srdf).getroot()
        for group in root.findall("group"):
            chain = group.find("chain")
            if chain is not None:
                return (
                    group.get("name"),
                    chain.get("base_link"),
                    chain.get("tip_link"),
                )
    except Exception:
        pass
    return kinova.MOVE_GROUP_ARM, "base_link", "end_effector_link"


def apply_scene(node: Node, scene_client, scene_msg: PlanningScene,
                timeout_s: float = 3.0) -> bool:
    req = ApplyPlanningScene.Request()
    req.scene = scene_msg
    future = scene_client.call_async(req)
    t0 = time.time()
    while rclpy.ok() and not future.done():
        if (time.time() - t0) > timeout_s:
            node.get_logger().error("apply_planning_scene timed out")
            return False
        time.sleep(0.01)
    try:
        resp = future.result()
        return bool(resp and resp.success)
    except Exception:
        return False


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
    return cfg


def _load_arm_joint_limits(moveit_config_pkg: str) -> Dict[str, Tuple[float, float]]:
    limits: Dict[str, Tuple[float, float]] = {}
    try:
        share = Path(get_package_share_directory(moveit_config_pkg))
        path = share / "config" / "joint_limits.yaml"
        data = yaml.safe_load(path.read_text()) or {}
        for joint_name, cfg in (data.get("joint_limits") or {}).items():
            if not cfg.get("has_position_limits", False):
                continue
            limits[joint_name] = (
                float(cfg["min_position"]),
                float(cfg["max_position"]),
            )
    except Exception:
        pass
    return limits


# ============================
# Supervisor Node
# ============================

class ClockPickSupervisorMetrics(Node):
    def __init__(self):
        super().__init__("clock_pick_supervisor_metrics")

        self.cb_group = ReentrantCallbackGroup()

        # ── Parameters ──
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("ee_frame", "tool_frame")
        self.declare_parameter("state_topic", "/clock_pick/state")
        self.declare_parameter("clock_marker_topic", "/clock/marker")
        self.declare_parameter("joint_state_topic", "/joint_states")

        self.declare_parameter("pose_timeout_s", 2.0)
        self.declare_parameter("detection_timeout_s", 30.0)
        self.declare_parameter("min_clock_conf", 0.40)

        self.declare_parameter("scan_poses", [])
        self.declare_parameter("scan_dwell_s", 5.0)
        self.declare_parameter("max_scan_cycles", 5)
        self.declare_parameter("enable_prescan", True)
        self.declare_parameter("prescan_dwell_sec", 0.40)
        self.declare_parameter("prescan_pose_settle_sec", 5.0)
        self.declare_parameter("scan_extra_pause_sec", 0.0)

        self.declare_parameter("moveit_max_vel", 0.15)
        self.declare_parameter("moveit_max_acc", 0.15)
        self.declare_parameter("planning_pipeline", "ompl")
        self.declare_parameter("planning_time_s", 5.0)
        self.declare_parameter("planning_attempts", 10)
        self.declare_parameter("robot_name", "kinova_gen3_6dof_robotiq_2f_85")
        self.declare_parameter("moveit_config_pkg", MOVEIT_PKG)
        self.declare_parameter("move_group_name", "manipulator")
        self.declare_parameter("try_pilz_lin", True)
        self.declare_parameter("pilz_namespace", "pilz_lin")
        self.declare_parameter("variant", "commit_only")
        self.declare_parameter("hold_commit_obstacle_pause_sec", 5.0)

        self.declare_parameter("gripper_open", 0.0)
        self.declare_parameter("gripper_closed", 0.35)
        self.declare_parameter("min_grasp_z_m", 0.03)

        self.declare_parameter("grasp_approach_angle_deg", 55.0)
        self.declare_parameter("yaw_offset_rad", 0.0)
        self.declare_parameter("finger_align_axis", "y")
        self.declare_parameter("grasp_end_inset_m", GRASP_END_INSET_M)
        self.declare_parameter("grasp_local_x_backoff_in", 16)
        self.declare_parameter("grasp_local_z_backoff_in", 5.0)
        self.declare_parameter("log_dir", "/tmp/clock_pick_logs")
        self.declare_parameter("global_summary_csv", "/tmp/clock_pick_metrics_summary.csv")
        self.declare_parameter("trial_id", "clock_pick_trial")

        self.declare_parameter("shutdown_after_done", True)
        self.declare_parameter("watchdog_timeout_s", 300.0)
        self.declare_parameter("start_state_clip_margin_rad", 1e-3)

        # Obstacle monitoring parameters
        self.declare_parameter("obstacle_gate_enabled", True)

        self.declare_parameter("collision_enabled", True)
        self.declare_parameter("table_point", [0.86, 0.0, -0.31645])
        self.declare_parameter("table_yaw", 0.0)
        self.declare_parameter("main_table_point", [-0.43056, 0.45, 0.13645])
        self.declare_parameter("main_table_yaw", math.pi / 2.0)
        self.declare_parameter("main_table_base_length_in", 22.0)
        self.declare_parameter("main_table_base_width_in", 60.0)
        self.declare_parameter("wheelchair_point", [-0.15, -0.39, 0.12065])
        self.declare_parameter("wheelchair_yaw", 0.0)
        self.declare_parameter("big_table_left_box_enabled", True)
        self.declare_parameter("big_table_left_box_length_in", 10.0)
        self.declare_parameter("big_table_left_box_width_in", 40.0)
        self.declare_parameter("big_table_left_box_height_in", 14.0)
        self.declare_parameter("big_table_left_box_gap_in", -13.0)
        self.declare_parameter("big_table_left_box_x_offset_in", -0.2)
        self.declare_parameter("big_table_left_box_yaw_deg", 90.0)

        self.declare_parameter("small_table_width_in", 17.0)
        self.declare_parameter("small_table_depth_in", 17.0)
        self.declare_parameter("small_table_height_in", 14.0)
        self.declare_parameter("small_table_side_height_in", 19.0)
        self.declare_parameter("small_table_wall_thickness_in", 1.0)
        self.declare_parameter("small_table_back_wall_full_height", True)
        self.declare_parameter("small_table_back_wall_height_in", 48.0)

        self.declare_parameter("obstacle_depth_topic", "/camera/depth/image_raw")
        self.declare_parameter("obstacle_roi_fraction", 0.30)
        self.declare_parameter("obstacle_min_valid_pixels", 200)
        self.declare_parameter("obstacle_percentile", 0.05)
        self.declare_parameter("obstacle_median_window", 5)
        self.declare_parameter("obstacle_stop_confirm_frames", 2)
        self.declare_parameter("obstacle_resume_confirm_frames", 2)
        self.declare_parameter("obstacle_depth_timeout_s", 0.75)
        self.declare_parameter("obstacle_fail_safe_blocked", True)
        self.declare_parameter("obstacle_stop_distance_m", 0.10)
        self.declare_parameter("obstacle_resume_distance_m", 0.13)
        self.declare_parameter("obstacle_ignore_object_within_m", 0.15)
        self.declare_parameter("obstacle_ignore_log_period_s", 1.0)
        self.declare_parameter("lin_approach_distance_m", 0.2)
        self.declare_parameter("lin_approach_obstacle_distance_m", 5.0 * INCH_TO_M)
        self.declare_parameter("lin_approach_step_m", 0.02)

        # ── Load parameters ──
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.ee_frame = str(self.get_parameter("ee_frame").value)
        self.state_topic = str(self.get_parameter("state_topic").value)

        self.clock_marker_topic = str(self.get_parameter("clock_marker_topic").value)
        self.joint_state_topic = str(self.get_parameter("joint_state_topic").value)

        self.pose_timeout_s = float(self.get_parameter("pose_timeout_s").value)
        self.detection_timeout_s = float(self.get_parameter("detection_timeout_s").value)
        self.min_clock_conf = float(self.get_parameter("min_clock_conf").value)

        self.scan_dwell_s = float(self.get_parameter("scan_dwell_s").value)
        self.max_scan_cycles = int(self.get_parameter("max_scan_cycles").value)
        self.enable_prescan = bool(self.get_parameter("enable_prescan").value)
        self.prescan_dwell_s = float(self.get_parameter("prescan_dwell_sec").value)
        self.prescan_pose_settle_s = float(self.get_parameter("prescan_pose_settle_sec").value)
        self.scan_extra_pause_s = float(self.get_parameter("scan_extra_pause_sec").value)

        self.moveit_max_vel = float(self.get_parameter("moveit_max_vel").value)
        self.moveit_max_acc = float(self.get_parameter("moveit_max_acc").value)
        self.planning_pipeline = str(self.get_parameter("planning_pipeline").value)
        self.planning_time_s = float(self.get_parameter("planning_time_s").value)
        self.planning_attempts = int(self.get_parameter("planning_attempts").value)
        self.robot_name = str(self.get_parameter("robot_name").value)
        self.moveit_config_pkg = str(self.get_parameter("moveit_config_pkg").value)
        self.move_group_name = str(self.get_parameter("move_group_name").value)
        self.try_pilz_lin = bool(self.get_parameter("try_pilz_lin").value)
        self.pilz_namespace = str(self.get_parameter("pilz_namespace").value)

        self.variant = str(self.get_parameter("variant").value).strip().lower()
        if self.variant not in SUPPORTED_VARIANTS:
            self.get_logger().warn(
                f"Unsupported variant '{self.variant}', falling back to 'commit_only'")
            self.variant = "commit_only"
        self.hold_commit_obstacle_pause_sec = float(
            self.get_parameter("hold_commit_obstacle_pause_sec").value)

        self.gripper_open = float(self.get_parameter("gripper_open").value)
        self.gripper_closed = float(self.get_parameter("gripper_closed").value)
        self.min_grasp_z_m = float(self.get_parameter("min_grasp_z_m").value)

        self.grasp_approach_angle_deg = float(self.get_parameter("grasp_approach_angle_deg").value)
        self.yaw_offset_rad = float(self.get_parameter("yaw_offset_rad").value)
        self.finger_align_axis = str(self.get_parameter("finger_align_axis").value).strip().lower()
        if self.finger_align_axis not in ("x", "y"):
            self.get_logger().warn("finger_align_axis must be 'x' or 'y'; defaulting to 'y'")
            self.finger_align_axis = "y"
        self.grasp_end_inset_m = float(self.get_parameter("grasp_end_inset_m").value)
        self.grasp_local_x_backoff_m = float(
            self.get_parameter("grasp_local_x_backoff_in").value
        ) * INCH_TO_M
        self.grasp_local_z_backoff_m = float(
            self.get_parameter("grasp_local_z_backoff_in").value
        ) * INCH_TO_M
        self.log_dir = str(self.get_parameter("log_dir").value)
        self.global_summary_csv = str(self.get_parameter("global_summary_csv").value)
        self.trial_id = str(self.get_parameter("trial_id").value)

        self.shutdown_after_done = bool(self.get_parameter("shutdown_after_done").value)
        self.watchdog_timeout_s = float(self.get_parameter("watchdog_timeout_s").value)
        self.start_state_clip_margin_rad = float(
            self.get_parameter("start_state_clip_margin_rad").value
        )

        # Obstacle params
        self._obstacle_gate_enabled = bool(self.get_parameter("obstacle_gate_enabled").value)

        # Collision params
        self._collision_enabled = bool(self.get_parameter("collision_enabled").value)

        self._obstacle_depth_topic = str(self.get_parameter("obstacle_depth_topic").value)
        self._obstacle_roi_fraction = float(self.get_parameter("obstacle_roi_fraction").value)
        self._obstacle_min_valid_pixels = int(self.get_parameter("obstacle_min_valid_pixels").value)
        self._obstacle_percentile = float(self.get_parameter("obstacle_percentile").value)
        self._obstacle_median_window = int(self.get_parameter("obstacle_median_window").value)
        self._obstacle_stop_confirm_frames = int(self.get_parameter("obstacle_stop_confirm_frames").value)
        self._obstacle_resume_confirm_frames = int(self.get_parameter("obstacle_resume_confirm_frames").value)
        self._obstacle_depth_timeout_s = float(self.get_parameter("obstacle_depth_timeout_s").value)
        self._obstacle_fail_safe_blocked = bool(self.get_parameter("obstacle_fail_safe_blocked").value)
        self._obstacle_stop_distance_m = float(self.get_parameter("obstacle_stop_distance_m").value)
        self._obstacle_resume_distance_m = float(self.get_parameter("obstacle_resume_distance_m").value)
        self._obstacle_ignore_object_within_m = float(self.get_parameter("obstacle_ignore_object_within_m").value)
        self._obstacle_ignore_log_period_s = float(self.get_parameter("obstacle_ignore_log_period_s").value)
        self.lin_approach_distance_m = float(self.get_parameter("lin_approach_distance_m").value)
        self.lin_approach_obstacle_distance_m = float(
            self.get_parameter("lin_approach_obstacle_distance_m").value
        )
        self.lin_approach_step_m = float(self.get_parameter("lin_approach_step_m").value)

        # ── Default scan poses (6-DoF Kinova Gen3) ──
        self.scan_poses_raw = list(self.get_parameter("scan_poses").value)
        if not self.scan_poses_raw:
            self.scan_poses_raw = [
                [0.6427, 0.9756, 1.4850, 0.2295, 1.2395, -1.6416],
                [0.5322, 0.8251, 1.2608, 0.1999, 1.6878, -1.6803],
                [1.5701, 0.7471, 1.3472, 0.1524, 1.2910, -1.6665],
                [1.2839, 0.8440, 1.3817, 0.2653, 1.4562, -1.6168],
            ]

        # ── ROS interfaces ──
        self.state_pub = self.create_publisher(String, self.state_topic, 10)
        self.ee_target_pose_pub = self.create_publisher(PoseStamped, "/clock_pick/ee_target_pose", 10)
        self.ee_staging_pose_pub = self.create_publisher(PoseStamped, "/clock_pick/ee_staging_pose", 10)
        self.ee_intermediate_pose_pub = self.create_publisher(PoseStamped, "/clock_pick/ee_intermediate_pose", 10)

        # TF (for obstacle ignore heuristic — EE distance check)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── Subscribe to clock marker from detection node ──
        self._pose_lock = threading.Lock()
        self._latest_clock_pose: Optional[Tuple[float, float, float, float]] = None  # (x,y,z,yaw)
        self._pose_stamp: float = 0.0
        self._pose_ok_current = False
        self._pose_ok_true_since = None
        self._clock_found_scan_idx = 0
        self._locked_clock_pose: Optional[Tuple[float, float, float, float]] = None
        self._allow_pose_updates = True

        self.create_subscription(
            Marker, self.clock_marker_topic,
            self._on_clock_marker, 10,
            callback_group=self.cb_group)

        self._latest_joint_positions: Dict[str, float] = {}
        self._joint_state_stamp = 0.0
        self._joint_limits = _load_arm_joint_limits(self.moveit_config_pkg)
        self.create_subscription(
            JointState,
            self.joint_state_topic,
            self._on_joint_state,
            20,
            callback_group=self.cb_group,
        )

        # ── Obstacle monitoring subscription ──
        self._obstacle_bridge = CvBridge()
        self._obstacle_last_depth_mono = None
        self._obstacle_last_depth_t = 0.0
        self._obstacle_blocked = False
        self._obstacle_gate_active = False
        self._obstacle_stop_count = 0
        self._obstacle_resume_count = 0
        self._obstacle_depth_history = []
        self._obstacle_ignore_last_log_mono = 0.0
        self._active_object_pose = None  

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.create_subscription(
            Image, self._obstacle_depth_topic,
            self._on_obstacle_depth_frame, qos,
            callback_group=self.cb_group)

        self.scene_client = self.create_client(
            ApplyPlanningScene, "/apply_planning_scene",
            callback_group=self.cb_group)
        while not self.scene_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /apply_planning_scene...")

        # ── MoveIt setup (fan-style MoveItPy) ──
        group_name, srdf_base_link, srdf_tip_link = infer_srdf()
        if self.base_frame == "base_link":
            self.base_frame = srdf_base_link
        if self.ee_frame == "tool_frame":
            self.ee_frame = srdf_tip_link
        self._group_name = group_name if not self.move_group_name else self.move_group_name
        self._moveit = None
        self._pc = None
        self._pilz_params = None
        self._default_planning_pipeline = self.planning_pipeline or "ompl"
        try:
            mcfg = MoveItConfigsBuilder(
                self.robot_name, package_name=self.moveit_config_pkg
            ).to_moveit_configs()
            cd = _augment_moveit_cfg(mcfg.to_dict(), jst="/joint_states")
            self._default_planning_pipeline = str(
                cd.get("default_planning_pipeline", self._default_planning_pipeline)
            )
            self._moveit = MoveItPy(node_name="clock_pick_moveitpy", config_dict=cd)
            self._pc = self._moveit.get_planning_component(self._group_name)
            self.get_logger().info(
                f"MoveItPy ready ({self.robot_name} / {self._group_name}) "
                f"default_pipeline={self._default_planning_pipeline} "
                f"vel={self.moveit_max_vel:.0%} acc={self.moveit_max_acc:.0%}"
            )
            if self.try_pilz_lin:
                try:
                    self._pilz_params = PlanRequestParameters(self._moveit, self.pilz_namespace)
                    self._pilz_params.planning_pipeline = "pilz_industrial_motion_planner"
                    self._pilz_params.planner_id = "LIN"
                    self._pilz_params.planning_time = min(2.0, self.planning_time_s)
                    self._pilz_params.planning_attempts = 1
                    self._pilz_params.max_velocity_scaling_factor = min(self.moveit_max_vel, 0.08)
                    self._pilz_params.max_acceleration_scaling_factor = min(self.moveit_max_acc, 0.08)
                except Exception as e:
                    self.get_logger().warn(f"Pilz init failed: {e}")
                    self._pilz_params = None
        except Exception as e:
            self.get_logger().error(f"MoveItPy init failed: {e}")
            self._request_shutdown("moveit_init_failed")

        self._scene_ready = False
        self._scene_attempts = 0
        self._scene_max_attempts = 10
        self._scene_retry_period_s = 2.0
        self._scene_last_try = 0.0

        # ── Gripper ──
        self.gripper = Gripper(self)
        self._gripper_ready = self.gripper.wait_for_server(5.0)

        # ── CSV / metrics ──
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        self.attempts_csv_path = os.path.join(self.log_dir, "attempts.csv")
        self.state_stream_csv_path = os.path.join(self.log_dir, "state_stream.csv")
        self.metrics_trial_csv_path = os.path.join(self.log_dir, "metrics_trial.csv")
        self.global_summary_csv_path = self.global_summary_csv

        self._init_csv()
        self._init_global_summary()

        self._trial_start_mono = time.time()
        self._started_flag = False
        self.t_start_s = None
        self.t_detect_s = None
        self.t_grasp_s = None
        self.t_release_s = None

        self._task_thread = None
        self._task_result = None
        self._shutdown_requested = False
        self._shutdown_at = None

        self._state = "INIT"
        self._publish_state()
        self._log_state_stream()
        self.get_logger().info(f"[DETECT] using marker topic: {self.clock_marker_topic}")

        self.last_grasp_quat = None
        self._hold_commit_paused_once = False  

        # ── Tick loop ──
        self._tick_timer = self.create_timer(0.1, self._tick, callback_group=self.cb_group)

    # ────────────────────── POSE / QUALITY CALLBACKS ──────────────────────

    def _on_clock_marker(self, msg: Marker):
        """Receive clock marker from either clock_pose_marker or clock_pose_node_metrics."""
        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)
        z = float(msg.pose.position.z)
        quat = [
            float(msg.pose.orientation.x),
            float(msg.pose.orientation.y),
            float(msg.pose.orientation.z),
            float(msg.pose.orientation.w),
        ]
        yaw = float(R.from_quat(quat).as_euler("xyz", degrees=False)[2])

        with self._pose_lock:
            if self._locked_clock_pose is not None and not self._allow_pose_updates:
                return
            self._latest_clock_pose = (x, y, z, yaw)
            self._pose_stamp = time.time()

    def _on_joint_state(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            if name in ARM_JOINTS:
                self._latest_joint_positions[name] = float(pos)
        self._joint_state_stamp = time.time()

    def _get_pose_if_fresh(self, max_age_s: float) -> Optional[Tuple[float, float, float, float]]:
        with self._pose_lock:
            if self._locked_clock_pose is not None and not self._allow_pose_updates:
                return self._locked_clock_pose
            if self._latest_clock_pose is None:
                return None
            age = time.time() - self._pose_stamp
            if age > max_age_s:
                return None
            return self._latest_clock_pose

    def _stable_pose(self, dwell_s: float, max_age_s: float) -> Optional[Tuple[float, float, float, float]]:
        pose = self._get_pose_if_fresh(max_age_s)
        now = self.get_clock().now()
        fresh = pose is not None
        if fresh and not self._pose_ok_current:
            self._pose_ok_true_since = now
        elif not fresh and self._pose_ok_current:
            self._pose_ok_true_since = None
        self._pose_ok_current = fresh
        if not fresh or self._pose_ok_true_since is None:
            return None
        if (now - self._pose_ok_true_since) >= Duration(seconds=float(dwell_s)):
            return pose
        return None

    def _lock_clock_pose(self, pose: Tuple[float, float, float, float], stage: str):
        with self._pose_lock:
            self._locked_clock_pose = tuple(float(v) for v in pose)
            self._latest_clock_pose = self._locked_clock_pose
            self._pose_stamp = time.time()
            self._allow_pose_updates = False
        self.get_logger().info(
            f"[DETECT] locked {stage} clock pose at "
            f"({pose[0]:.3f},{pose[1]:.3f},{pose[2]:.3f},y={pose[3]:.2f})"
        )

    def _set_start_state_from_joint_cache(self) -> bool:
        if self._pc is None:
            return False
        if not all(j in self._latest_joint_positions for j in ARM_JOINTS):
            return False

        positions = []
        clipped = []
        margin = max(0.0, self.start_state_clip_margin_rad)
        for joint_name in ARM_JOINTS:
            pos = float(self._latest_joint_positions[joint_name])
            limits = self._joint_limits.get(joint_name)
            if limits is not None:
                lo, hi = limits
                clipped_pos = min(max(pos, lo + margin), hi - margin)
                if abs(clipped_pos - pos) > 1e-9:
                    clipped.append(f"{joint_name}:{pos:.5f}->{clipped_pos:.5f}")
                pos = clipped_pos
            positions.append(pos)

        if clipped:
            self.get_logger().warn(
                "[MOVE] clipped start-state joints to bounds: " + ", ".join(clipped)
            )

        start_state = RobotState(self._moveit.get_robot_model())
        start_state.set_joint_group_positions(self._group_name, positions)
        start_state.update()
        self._pc.set_start_state(robot_state=start_state)
        return True

    def _current_ee_pose(
        self, timeout_s: float = 0.2
    ) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]]:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.ee_frame,
                Time(),
                timeout=Duration(seconds=timeout_s),
            )
        except Exception:
            return None
        pos = (
            float(tf.transform.translation.x),
            float(tf.transform.translation.y),
            float(tf.transform.translation.z),
        )
        quat = (
            float(tf.transform.rotation.x),
            float(tf.transform.rotation.y),
            float(tf.transform.rotation.z),
            float(tf.transform.rotation.w),
        )
        return pos, quat

    def _compute_clock_relative_grasp_quat(
        self, grasp_target: List[float], yaw: float
    ) -> Optional[Tuple[List[float], List[float], List[float], List[float]]]:
        ee_pose = self._current_ee_pose(timeout_s=0.3)
        if ee_pose is None:
            return None
        ee_pos, _ = ee_pose

        x_tool = np.array([-math.sin(yaw), math.cos(yaw), 0.0], dtype=float)
        x_norm = float(np.linalg.norm(x_tool))
        if x_norm < 1e-9:
            return None
        x_tool = x_tool / x_norm

        target_vec = np.array(grasp_target, dtype=float) - np.array(ee_pos, dtype=float)
        z_tool = target_vec - float(np.dot(target_vec, x_tool)) * x_tool
        z_norm = float(np.linalg.norm(z_tool))
        if z_norm < 1e-9:
            return None
        z_tool = z_tool / z_norm

        y_tool = np.cross(z_tool, x_tool)
        y_norm = float(np.linalg.norm(y_tool))
        if y_norm < 1e-9:
            return None
        y_tool = y_tool / y_norm

        if y_tool[2] < 0.0:
            y_tool = -y_tool
            z_tool = -z_tool

        z_tool = np.cross(x_tool, y_tool)
        z_tool = z_tool / max(1e-9, float(np.linalg.norm(z_tool)))

        quat = R.from_matrix(np.column_stack((x_tool, y_tool, z_tool))).as_quat().tolist()
        return quat, x_tool.tolist(), y_tool.tolist(), z_tool.tolist()

    # ────────────────────── CSV / METRICS ──────────────────────

    def _init_csv(self):
        if not os.path.exists(self.attempts_csv_path):
            with open(self.attempts_csv_path, "w") as f:
                f.write("trial_id,variant,started_flag,t_start_s,t_detect_s,t_grasp_s,t_release_s,outcome,detail\n")
        if not os.path.exists(self.state_stream_csv_path):
            with open(self.state_stream_csv_path, "w") as f:
                f.write("t_s,trial_id,variant,state\n")
        if not os.path.exists(self.metrics_trial_csv_path):
            with open(self.metrics_trial_csv_path, "w") as f:
                f.write("trial_id,variant,started_flag,t_start_s,t_detect_s,t_grasp_s,t_release_s,success,detail\n")

    def _init_global_summary(self):
        if not os.path.exists(self.global_summary_csv_path):
            with open(self.global_summary_csv_path, "w") as f:
                f.write("trial_id,variant,started_flag,t_start_s,t_detect_s,t_grasp_s,t_release_s,success,detail\n")

    def _append_csv(self, path: str, row: str):
        with open(path, "a") as f:
            f.write(row + "\n")

    def _trial_t(self) -> float:
        return time.time() - self._trial_start_mono

    def _mark_started(self):
        if not self._started_flag:
            self._started_flag = True
            self.t_start_s = self._trial_t()

    def _write_metrics_files(self, success: bool, detail: str):
        row = (
            f"{self.trial_id},{self.variant},{int(self._started_flag)},"
            f"{'' if self.t_start_s is None else f'{self.t_start_s:.3f}'},"
            f"{'' if self.t_detect_s is None else f'{self.t_detect_s:.3f}'},"
            f"{'' if self.t_grasp_s is None else f'{self.t_grasp_s:.3f}'},"
            f"{'' if self.t_release_s is None else f'{self.t_release_s:.3f}'},"
            f"{int(success)},{detail}"
        )
        self._append_csv(self.metrics_trial_csv_path, row)
        self._append_csv(self.global_summary_csv_path, row)

        attempts_row = (
            f"{self.trial_id},{self.variant},{int(self._started_flag)},"
            f"{'' if self.t_start_s is None else f'{self.t_start_s:.3f}'},"
            f"{'' if self.t_detect_s is None else f'{self.t_detect_s:.3f}'},"
            f"{'' if self.t_grasp_s is None else f'{self.t_grasp_s:.3f}'},"
            f"{'' if self.t_release_s is None else f'{self.t_release_s:.3f}'},"
            f"{'success' if success else 'fail'},{detail}"
        )
        self._append_csv(self.attempts_csv_path, attempts_row)

    def _finish(self, success: bool, detail: str):
        self._write_metrics_files(success, detail)
        if self.shutdown_after_done:
            self._request_shutdown("task_complete")

    # ────────────────────── STATE MACHINE ──────────────────────

    def _publish_state(self):
        msg = String()
        msg.data = self._state
        self.state_pub.publish(msg)

    def _publish_target_pose(
        self, pub, xyz: List[float], quat_xyzw: List[float]
    ):
        msg = PoseStamped()
        msg.header.frame_id = self.base_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(xyz[0])
        msg.pose.position.y = float(xyz[1])
        msg.pose.position.z = float(xyz[2])
        msg.pose.orientation.x = float(quat_xyzw[0])
        msg.pose.orientation.y = float(quat_xyzw[1])
        msg.pose.orientation.z = float(quat_xyzw[2])
        msg.pose.orientation.w = float(quat_xyzw[3])
        pub.publish(msg)

    def _set_state(self, state: str):
        if state != self._state:
            self.get_logger().info(f"[STATE] {self._state} -> {state}")
            self._state = state
            self._publish_state()
            self._log_state_stream()

    def _log_state_stream(self):
        row = f"{self._trial_t():.3f},{self.trial_id},{self.variant},{self._state}"
        self._append_csv(self.state_stream_csv_path, row)

    # ────────────────────── SHUTDOWN ──────────────────────

    def _request_shutdown(self, reason: str):
        if not self._shutdown_requested:
            self._shutdown_requested = True
            self._shutdown_at = time.time()
            self.get_logger().info(f"[SHUTDOWN] requested: {reason}")

    def _shutdown_now(self):
        self._shutdown_at = None
        try:
            self.gripper.cancel()
        except Exception:
            pass
        self.get_logger().info("[SHUTDOWN] exiting node")
        if rclpy.ok():
            rclpy.shutdown()

    def _watchdog_timeout(self) -> bool:
        return self._trial_t() > self.watchdog_timeout_s

    # ────────────────────── TICK TIMER ──────────────────────

    def _tick(self):
        now = time.time()

        if self._watchdog_timeout():
            self.get_logger().error("[WATCHDOG] timeout exceeded")
            self._task_result = (False, "watchdog_timeout")
            self._finish(False, "watchdog_timeout")

        if self._shutdown_requested and self._shutdown_at is not None:
            if now - self._shutdown_at > 0.25:
                self._shutdown_now()
            return

        # Collision scene retry
        if not self._scene_ready:
            if ((now - self._scene_last_try) > self._scene_retry_period_s
                    and self._scene_attempts < self._scene_max_attempts):
                self._scene_last_try = now
                self._scene_attempts += 1
                self.get_logger().info(f"[SCENE] applying collision scene attempt {self._scene_attempts}")
                try:
                    self._scene_ready = bool(self._setup_collision_objects())
                    if self._scene_ready:
                        self.get_logger().info("[SCENE] collision scene ready")
                    else:
                        self.get_logger().warn("[SCENE] apply failed; will retry")
                except Exception as e:
                    self.get_logger().warn(f"[SCENE] apply failed: {e}")
            return

        # Launch task thread
        if self._task_thread is None:
            self._task_thread = threading.Thread(target=self._run_task, daemon=True)
            self._task_thread.start()
            return

        # Check result
        if self._task_result is not None:
            ok, msg = self._task_result
            self._finish(bool(ok), str(msg))
            self._task_result = None

    # ────────────────────── MOVEIT MOTION HELPERS ──────────────────────

    def _execute_trajectory(self, trajectory) -> bool:
        if self._moveit is None:
            return False
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

    def _log_motion_diag(self, diag: MotionDiag):
        self.get_logger().info(
            f"[MOVE][{diag.phase}] ok={diag.ok} cart={diag.cartesian} "
            f"t={diag.elapsed_s:.2f}s plan_t={diag.planning_time_s:.1f}s "
            f"attempts={diag.attempts} detail='{diag.detail}'"
        )

    def _plan_exec_joints(self, joints: List[float], planning_time: float) -> bool:
        if self._moveit is None or self._pc is None:
            return False
        try:
            if not self._set_start_state_from_joint_cache():
                self._pc.set_start_state_to_current_state()
            gs = RobotState(self._moveit.get_robot_model())
            gs.set_joint_group_positions(self._group_name, joints)
            gs.update()
            self._pc.set_goal_state(robot_state=gs)

            params = PlanRequestParameters(self._moveit, "")
            params.planning_pipeline = self.planning_pipeline or self._default_planning_pipeline
            params.planning_time = float(planning_time)
            params.planning_attempts = max(1, int(self.planning_attempts))
            params.max_velocity_scaling_factor = self.moveit_max_vel
            params.max_acceleration_scaling_factor = self.moveit_max_acc

            result = self._pc.plan(single_plan_parameters=params)
            if not result:
                return False
            return self._execute_trajectory(result.trajectory)
        except Exception as e:
            self.get_logger().warn(f"[MOVE] plan_exec_joints exception: {e}")
            return False

    def _plan_exec_pose(self, pose: PoseStamped, linear: bool, planning_time: float) -> bool:
        if self._moveit is None or self._pc is None:
            return False
        try:
            if not self._set_start_state_from_joint_cache():
                self._pc.set_start_state_to_current_state()
            self._pc.set_goal_state(pose_stamped_msg=pose, pose_link=self.ee_frame)

            if linear and self._pilz_params is not None:
                result = self._pc.plan(single_plan_parameters=self._pilz_params)
                if result:
                    return self._execute_trajectory(result.trajectory)

            params = PlanRequestParameters(self._moveit, "")
            params.planning_pipeline = self.planning_pipeline or self._default_planning_pipeline
            params.planning_time = float(planning_time)
            params.planning_attempts = max(1, int(self.planning_attempts))
            params.max_velocity_scaling_factor = self.moveit_max_vel
            params.max_acceleration_scaling_factor = self.moveit_max_acc

            result = self._pc.plan(single_plan_parameters=params)
            if not result:
                return False
            return self._execute_trajectory(result.trajectory)
        except Exception as e:
            self.get_logger().warn(f"[MOVE] plan_exec_pose exception: {e}")
            return False

    def _move_joints(self, joints: List[float], phase: str,
                     planning_time_s: Optional[float] = None,
                     attempts: Optional[int] = None) -> bool:
        pt = self.planning_time_s if planning_time_s is None else float(planning_time_s)
        att = self.planning_attempts if attempts is None else int(attempts)

        t0 = time.time()
        ok = self._plan_exec_joints(joints, planning_time=pt)
        dt = time.time() - t0
        self._log_motion_diag(MotionDiag(
            phase=phase, ok=ok, cartesian=False,
            planning_time_s=pt, attempts=att,
            elapsed_s=dt, detail="joints"))
        return ok

    def _move_pose(self, target_xyz: List[float], target_quat_xyzw: List[float],
                   phase: str, cartesian: bool,
                   planning_time_s: Optional[float] = None,
                   attempts: Optional[int] = None) -> bool:
        pt = self.planning_time_s if planning_time_s is None else float(planning_time_s)
        att = self.planning_attempts if attempts is None else int(attempts)

        pose = PoseStamped()
        pose.header.frame_id = self.base_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(target_xyz[0])
        pose.pose.position.y = float(target_xyz[1])
        pose.pose.position.z = float(target_xyz[2])
        pose.pose.orientation.x = float(target_quat_xyzw[0])
        pose.pose.orientation.y = float(target_quat_xyzw[1])
        pose.pose.orientation.z = float(target_quat_xyzw[2])
        pose.pose.orientation.w = float(target_quat_xyzw[3])

        t0 = time.time()
        ok = self._plan_exec_pose(pose, linear=cartesian, planning_time=pt)
        dt = time.time() - t0
        self._log_motion_diag(MotionDiag(
            phase=phase, ok=ok, cartesian=cartesian,
            planning_time_s=pt, attempts=att,
            elapsed_s=dt, detail="pose"))
        return ok

    # Quaternion math utilities
    def _quat_multiply(self, q1: List[float], q2: List[float]) -> List[float]:
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]

    def _rotate_vec_by_quat(self, v: List[float], q: List[float]) -> List[float]:
        q_conj = [-q[0], -q[1], -q[2], q[3]]
        vq = [v[0], v[1], v[2], 0.0]
        r = self._quat_multiply(self._quat_multiply(q, vq), q_conj)
        return [r[0], r[1], r[2]]

    # ────────────────────── COLLISION SCENE ──────────────────────

    @staticmethod
    def _make_box_prim(size: List[float]) -> SolidPrimitive:
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = list(size)
        return box

    @staticmethod
    def _make_collision_pose(position, yaw: float = 0.0) -> Pose:
        p = Pose()
        p.position.x = float(position[0])
        p.position.y = float(position[1])
        p.position.z = float(position[2])
        half = yaw / 2.0
        p.orientation.x = 0.0
        p.orientation.y = 0.0
        p.orientation.z = math.sin(half)
        p.orientation.w = math.cos(half)
        return p

    def _apply_base_to_local(self, base_pose: Pose, local_pose: Pose) -> Pose:
        """Transform a local pose into world frame given a base pose."""
        out = Pose()
        bq = [
            base_pose.orientation.x, base_pose.orientation.y,
            base_pose.orientation.z, base_pose.orientation.w,
        ]
        local_pos = [
            local_pose.position.x, local_pose.position.y,
            local_pose.position.z,
        ]
        rotated = self._rotate_vec_by_quat(local_pos, bq)
        out.position.x = base_pose.position.x + rotated[0]
        out.position.y = base_pose.position.y + rotated[1]
        out.position.z = base_pose.position.z + rotated[2]
        lq = [
            local_pose.orientation.x, local_pose.orientation.y,
            local_pose.orientation.z, local_pose.orientation.w,
        ]
        rx, ry, rz, rw = self._quat_multiply(bq, lq)
        out.orientation.x = rx
        out.orientation.y = ry
        out.orientation.z = rz
        out.orientation.w = rw
        return out

    def _build_collision_objects(self) -> List[CollisionObject]:
        """Build all collision objects matching the bottle task:
        main_table_shelf, table_shelf (small table), wheelchair."""

        # ── Read parameters ──
        table_point = list(self.get_parameter("table_point").value)
        table_yaw = float(self.get_parameter("table_yaw").value)
        main_table_point = list(self.get_parameter("main_table_point").value)
        main_table_yaw = float(self.get_parameter("main_table_yaw").value)
        main_table_base_length_m = float(self.get_parameter("main_table_base_length_in").value) * INCH_TO_M
        main_table_base_width_m = float(self.get_parameter("main_table_base_width_in").value) * INCH_TO_M
        wheelchair_point = list(self.get_parameter("wheelchair_point").value)
        wheelchair_yaw = float(self.get_parameter("wheelchair_yaw").value)

        big_left_box_enabled = bool(self.get_parameter("big_table_left_box_enabled").value)
        big_left_box_length_m = float(self.get_parameter("big_table_left_box_length_in").value) * INCH_TO_M
        big_left_box_width_m = float(self.get_parameter("big_table_left_box_width_in").value) * INCH_TO_M
        big_left_box_height_m = float(self.get_parameter("big_table_left_box_height_in").value) * INCH_TO_M
        big_left_box_gap_m = float(self.get_parameter("big_table_left_box_gap_in").value) * INCH_TO_M
        big_left_box_x_offset_m = float(self.get_parameter("big_table_left_box_x_offset_in").value) * INCH_TO_M
        big_left_box_yaw_rad = math.radians(float(self.get_parameter("big_table_left_box_yaw_deg").value))

        table_width_m = float(self.get_parameter("small_table_width_in").value) * INCH_TO_M
        table_depth_m = float(self.get_parameter("small_table_depth_in").value) * INCH_TO_M
        table_height_m = float(self.get_parameter("small_table_height_in").value) * INCH_TO_M
        side_height_m = float(self.get_parameter("small_table_side_height_in").value) * INCH_TO_M
        side_thickness_m = float(self.get_parameter("small_table_wall_thickness_in").value) * INCH_TO_M
        side_thickness_m = max(0.005, side_thickness_m)
        back_wall_full_height = bool(self.get_parameter("small_table_back_wall_full_height").value)
        back_wall_height_m = float(self.get_parameter("small_table_back_wall_height_in").value) * INCH_TO_M
        side_extension_m = max(0.0, side_height_m - table_height_m)

        base_table_pose = self._make_collision_pose(table_point, table_yaw)
        base_main_table_pose = self._make_collision_pose(main_table_point, main_table_yaw)
        base_wheel_pose = self._make_collision_pose(wheelchair_point, wheelchair_yaw)

        # ══════════════════════════════════════════════════════════════
        # 1. main_table_shelf — big table + shelf unit + optional left box
        # ══════════════════════════════════════════════════════════════
        full_table_size = [main_table_base_length_m, main_table_base_width_m, 0.2921 + 0.0127]
        table_abs_pose = self._make_collision_pose((0.6985, 0.4556, 0.14605))

        shelf_outer_size = [0.3048, 1.1176, 0.5842]
        table_edge_to_shelf = 0.4572 - 0.0127
        shelf_origin_x = 0.3175 + table_edge_to_shelf
        shelf_origin_y = 0.0 - 0.025
        shelf_origin_z = full_table_size[2] + 0.0127
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

        full_table_prims = []

        # Table surface
        full_table_prims.append((self._make_box_prim(full_table_size), table_abs_pose))

        # 3 horizontal shelf planes
        for z_rel in shelf_heights:
            z = shelf_origin_z + z_rel + shelf_thickness / 2.0
            full_table_prims.append((
                self._make_box_prim([
                    shelf_outer_size[0], shelf_outer_size[1], shelf_thickness]),
                self._make_collision_pose((
                    shelf_origin_x + shelf_outer_size[0] / 2.0, 0.0, z-0.01)),
            ))

        # Left side wall
        full_table_prims.append((
            self._make_box_prim([
                shelf_outer_size[0], 0.02, shelf_outer_size[2]]),
            self._make_collision_pose((
                shelf_origin_x + shelf_outer_size[0] / 2.0,
                shelf_origin_y - shelf_outer_size[1] / 2.0,
                shelf_origin_z + shelf_outer_size[2] / 2.0)),
        ))

        # Right side wall
        full_table_prims.append((
            self._make_box_prim([
                shelf_outer_size[0], 0.02, shelf_outer_size[2]]),
            self._make_collision_pose((
                shelf_origin_x + shelf_outer_size[0] / 2.0,
                shelf_origin_y + shelf_outer_size[1] / 2.0,
                shelf_origin_z + shelf_outer_size[2] / 2.0)),
        ))

        # Back wall
        full_table_prims.append((
            self._make_box_prim([
                0.02, shelf_outer_size[1], shelf_outer_size[2]]),
            self._make_collision_pose((
                shelf_origin_x + shelf_outer_size[0] - 0.01,
                0.0,
                shelf_origin_z + shelf_outer_size[2] / 2.0)),
        ))

        # Front partial wall (bottom section)
        full_table_prims.append((
            self._make_box_prim([
                0.02, shelf_outer_size[1],
                bottom_to_shelf1 - half_shelf_thickness]),
            self._make_collision_pose((
                shelf_origin_x, 0.0,
                shelf_origin_z + bottom_to_shelf1 / 2.0)),
        ))

        # Vertical divider (shelf 1 to shelf 2) — left partition
        middle_to_part = (shelf_outer_size[1] / 2.0) - left_to_part1
        full_table_prims.append((
            self._make_box_prim([
                shelf_outer_size[0], 0.02, shelf1_to_shelf2]),
            self._make_collision_pose((
                shelf_origin_x + shelf_outer_size[0] / 2.0,
                middle_to_part,
                shelf_origin_z + shelf1_to_shelf2)),
        ))

        # Vertical divider (shelf 1 to shelf 2) — right partition
        full_table_prims.append((
            self._make_box_prim([
                shelf_outer_size[0], 0.02, shelf1_to_shelf2]),
            self._make_collision_pose((
                shelf_origin_x + shelf_outer_size[0] / 2.0,
                -middle_to_part,
                shelf_origin_z + shelf1_to_shelf2)),
        ))

        # Center divider (shelf 2 to top)
        full_table_prims.append((
            self._make_box_prim([
                shelf_outer_size[0], 0.02, shelf2_to_top]),
            self._make_collision_pose((
                shelf_origin_x + shelf_outer_size[0] / 2.0,
                -0.1,
                shelf_origin_z + bottom_to_shelf1
                + shelf2_to_top - shelf_thickness / 2.0)),
        ))

        # Optional big left box obstacle on the left side of the shelf
        if big_left_box_enabled:
            box_length_m = max(0.01, big_left_box_length_m)
            box_width_m = max(0.01, big_left_box_width_m)
            box_height_m = max(0.01, big_left_box_height_m)
            left_shelf_edge_y = shelf_origin_y + (shelf_outer_size[1] / 2.0)
            box_center_x = shelf_origin_x + (shelf_outer_size[0] / 2.0) + big_left_box_x_offset_m
            box_center_y = left_shelf_edge_y + (box_width_m / 2.0) + big_left_box_gap_m
            box_center_z = full_table_size[2] + (box_height_m / 2.0)
            full_table_prims.append((
                self._make_box_prim([box_length_m, box_width_m, box_height_m]),
                self._make_collision_pose(
                    (box_center_x, box_center_y, box_center_z),
                    big_left_box_yaw_rad),
            ))

        # Assemble main_table_shelf CollisionObject
        co_main_table = CollisionObject()
        co_main_table.id = "main_table_shelf"
        co_main_table.header.frame_id = self.base_frame
        co_main_table.operation = CollisionObject.ADD

        for prim, abs_pose in full_table_prims:
            local = Pose()
            local.position.x = abs_pose.position.x - table_ref[0]
            local.position.y = abs_pose.position.y - table_ref[1]
            local.position.z = abs_pose.position.z - table_ref[2]
            local.orientation = abs_pose.orientation
            world_pose = self._apply_base_to_local(base_main_table_pose, local)
            co_main_table.primitives.append(prim)
            co_main_table.primitive_poses.append(world_pose)

        # ══════════════════════════════════════════════════════════════
        # 2. table_shelf — small table with side lips and back wall
        # ══════════════════════════════════════════════════════════════
        co_table = CollisionObject()
        co_table.id = "table_shelf"
        co_table.header.frame_id = self.base_frame
        co_table.operation = CollisionObject.ADD

        # Table top slab
        local = Pose()
        local.orientation.w = 1.0
        co_table.primitives.append(
            self._make_box_prim([table_depth_m, table_width_m, table_height_m]))
        co_table.primitive_poses.append(
            self._apply_base_to_local(base_table_pose, local))

        # Side lip extensions (raised edges)
        if side_extension_m > 1e-6:
            lip_center_z = (table_height_m / 2.0) + (side_extension_m / 2.0)

            # Left lip
            left = Pose()
            left.position.y = (table_width_m / 2.0) - (side_thickness_m / 2.0)
            left.position.z = lip_center_z
            left.orientation.w = 1.0
            co_table.primitives.append(
                self._make_box_prim([table_depth_m, side_thickness_m, side_extension_m]))
            co_table.primitive_poses.append(
                self._apply_base_to_local(base_table_pose, left))

            # Right lip
            right = Pose()
            right.position.y = -((table_width_m / 2.0) - (side_thickness_m / 2.0))
            right.position.z = lip_center_z
            right.orientation.w = 1.0
            co_table.primitives.append(
                self._make_box_prim([table_depth_m, side_thickness_m, side_extension_m]))
            co_table.primitive_poses.append(
                self._apply_base_to_local(base_table_pose, right))

        # Back wall
        back_height_use = back_wall_height_m if back_wall_full_height else side_extension_m
        if back_height_use > 1e-6:
            back = Pose()
            back.position.x = (table_depth_m / 2.0) - (side_thickness_m / 2.0)
            back.position.z = back_height_use / 2.0
            back.orientation.w = 1.0
            co_table.primitives.append(
                self._make_box_prim([side_thickness_m, table_width_m, back_height_use]))
            co_table.primitive_poses.append(
                self._apply_base_to_local(base_table_pose, back))

        # ══════════════════════════════════════════════════════════════
        # 3. wheelchair
        # ══════════════════════════════════════════════════════════════
        wall_size = [0.51, 0.5, 0.2413]
        co_wheel = CollisionObject()
        co_wheel.id = "wheelchair"
        co_wheel.header.frame_id = self.base_frame
        co_wheel.operation = CollisionObject.ADD

        local = Pose()
        local.orientation.w = 1.0
        world_pose = self._apply_base_to_local(base_wheel_pose, local)
        co_wheel.primitives.append(self._make_box_prim(wall_size))
        co_wheel.primitive_poses.append(world_pose)

        return [co_main_table, co_table, co_wheel]

    def _setup_collision_objects(self):
        if not self._collision_enabled:
            return True
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = self._build_collision_objects()
        ok = apply_scene(self, self.scene_client, scene, timeout_s=3.0)
        ids = [co.id for co in scene.world.collision_objects]
        counts = [len(co.primitives) for co in scene.world.collision_objects]
        if ok:
            self.get_logger().info(
                f"[SCENE] published {len(ids)} objects: "
                + ", ".join(f"'{i}' ({c})" for i, c in zip(ids, counts)))
        return ok

    # ────────────────────── OBSTACLE MONITORING ──────────────────────

    def _set_obstacle_gate(self, active: bool, reason: str):
        self._obstacle_gate_active = bool(active)
        self._obstacle_stop_count = 0
        self._obstacle_resume_count = 0
        self.get_logger().info(f"[OBST] gate={'ON' if active else 'OFF'} reason={reason}")

    def _gripper_to_active_object_distance(self) -> Optional[float]:
        if self._active_object_pose is None:
            return None
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, self.ee_frame, Time(),
                timeout=rclpy.duration.Duration(seconds=0.1))
            ex = float(tf.transform.translation.x)
            ey = float(tf.transform.translation.y)
            ez = float(tf.transform.translation.z)
        except Exception:
            return None
        bx, by, bz = self._active_object_pose
        return float(math.sqrt((ex - bx)**2 + (ey - by)**2 + (ez - bz)**2))

    def _should_ignore_obstacle_for_object(self) -> bool:
        dist = self._gripper_to_active_object_distance()
        if dist is None:
            return False
        if dist < self._obstacle_ignore_object_within_m:
            now = time.time()
            if now - self._obstacle_ignore_last_log_mono > self._obstacle_ignore_log_period_s:
                self._obstacle_ignore_last_log_mono = now
                self.get_logger().info(
                    f"[OBST] ignoring obstacle (object near gripper): dist={dist:.3f}m")
            return True
        return False

    def _on_obstacle_depth_frame(self, msg: Image):
        try:
            depth = self._obstacle_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception:
            return

        h, w = depth.shape[:2]
        roi_w = max(1, int(w * self._obstacle_roi_fraction))
        roi_h = max(1, int(h * self._obstacle_roi_fraction))
        x0 = (w - roi_w) // 2
        y0 = (h - roi_h) // 2
        roi = depth[y0:y0 + roi_h, x0:x0 + roi_w].astype(np.float32)

        vals = roi[np.isfinite(roi)]
        vals = vals[vals > 0.0]
        if len(vals) < self._obstacle_min_valid_pixels:
            self._obstacle_last_depth_mono = None
            self._obstacle_last_depth_t = time.time()
            if self._obstacle_fail_safe_blocked:
                self._obstacle_blocked = True
            return

        d = float(np.percentile(vals, self._obstacle_percentile * 100.0))
        if d > 100.0:
            d /= 1000.0

        self._obstacle_depth_history.append(d)
        if len(self._obstacle_depth_history) > self._obstacle_median_window:
            self._obstacle_depth_history.pop(0)
        d_med = float(np.median(self._obstacle_depth_history))

        self._obstacle_last_depth_mono = d_med
        self._obstacle_last_depth_t = time.time()

        if self._should_ignore_obstacle_for_object():
            return

        if not self._obstacle_blocked:
            if d_med < self._obstacle_stop_distance_m:
                self._obstacle_stop_count += 1
            else:
                self._obstacle_stop_count = 0
            if self._obstacle_stop_count >= self._obstacle_stop_confirm_frames:
                self._obstacle_blocked = True
                self._obstacle_resume_count = 0
        else:
            if d_med > self._obstacle_resume_distance_m:
                self._obstacle_resume_count += 1
            else:
                self._obstacle_resume_count = 0
            if self._obstacle_resume_count >= self._obstacle_resume_confirm_frames:
                self._obstacle_blocked = False
                self._obstacle_stop_count = 0

    def _obstacle_is_blocked(self) -> bool:
        if not self._obstacle_gate_enabled or not self._obstacle_gate_active:
            return False
        if time.time() - self._obstacle_last_depth_t > self._obstacle_depth_timeout_s:
            return bool(self._obstacle_fail_safe_blocked)
        return bool(self._obstacle_blocked)

    # ────────────────────── OBSTACLE-AWARE MOTION ──────────────────────

    def _obstacle_aware_move(self, target_xyz: List[float], target_quat: List[float],
                             phase: str, cartesian: bool) -> bool:
        while True:
            if self._obstacle_is_blocked():
                if self.variant == "commit_only":
                    self.get_logger().warn(
                        f"[OBST] blocked during {phase} (commit_only): not retrying")
                    return False

                if self.variant == "feas_only":
                    self.get_logger().error(
                        f"[OBST] blocked during {phase} (feas_only): aborting task")
                    self._task_result = (False, f"feas_only abort: obstacle during {phase}")
                    raise _ObstacleAbort()

                if self.variant == "hold_commit":
                    if not self._hold_commit_paused_once:
                        self.get_logger().warn(
                            f"[OBST] blocked during {phase} (hold_commit): "
                            f"pausing {self.hold_commit_obstacle_pause_sec:.1f}s then retry")
                        time.sleep(self.hold_commit_obstacle_pause_sec)
                        self._hold_commit_paused_once = True
                    else:
                        self.get_logger().warn(
                            f"[OBST] blocked during {phase} (hold_commit): "
                            f"already paused once, retrying immediately")
                    return self._move_pose(target_xyz, target_quat, phase, cartesian=cartesian)

                if self.variant == "hac":
                    self.get_logger().warn(
                        f"[OBST] blocked during {phase} (hac): waiting to clear")
                    while self._obstacle_is_blocked() and rclpy.ok():
                        time.sleep(0.10)
                    self.get_logger().info(f"[OBST] cleared; retrying {phase}")
                    return self._move_pose(target_xyz, target_quat, phase, cartesian=cartesian)

            ok = self._move_pose(target_xyz, target_quat, phase, cartesian=cartesian)
            return ok

    # ────────────────────── LIN APPROACH (pre-grasp probe) ──────────────────────

    def _lin_approach_is_blocked(self, threshold_m: float) -> bool:
        if not self._obstacle_gate_enabled:
            return False
        if time.time() - self._obstacle_last_depth_t > self._obstacle_depth_timeout_s:
            return bool(self._obstacle_fail_safe_blocked)
        d = self._obstacle_last_depth_mono
        if d is None:
            return bool(self._obstacle_fail_safe_blocked)
        return float(d) <= float(threshold_m)

    def _reverse_lin_approach(
        self, executed_waypoints: List[List[float]], ee_quat_xyzw: List[float]
    ) -> None:
        if len(executed_waypoints) <= 1:
            return
        for idx, xyz in enumerate(reversed(executed_waypoints[:-1]), start=1):
            ok = self._move_pose(
                xyz, ee_quat_xyzw, phase=f"lin_approach_reverse_{idx}", cartesian=True
            )
            if not ok:
                self.get_logger().warn(
                    "[LIN_APPROACH] reverse segment failed; trying direct return to start"
                )
                start_xyz = executed_waypoints[0]
                direct_ok = self._move_pose(
                    start_xyz, ee_quat_xyzw, phase="lin_approach_reverse_start", cartesian=True
                )
                if not direct_ok:
                    self._move_pose(
                        start_xyz,
                        ee_quat_xyzw,
                        phase="lin_approach_reverse_start_jt",
                        cartesian=False,
                    )
                return

    def _execute_lin_approach(self, grasp_target: List[float],
                              approach_direction: Optional[List[float]] = None) -> None:
        ee_pose = self._current_ee_pose(timeout_s=0.5)
        if ee_pose is None:
            self.get_logger().warn("[LIN_APPROACH] EE pose unavailable; skipping")
            return

        ee_pos, ee_quat = ee_pose
        ee_np = np.array(ee_pos, dtype=float)
        target_np = np.array(grasp_target, dtype=float)

        if approach_direction is not None:
            direction = np.array(approach_direction, dtype=float)
            d_norm = float(np.linalg.norm(direction))
            if d_norm < 1e-6:
                self.get_logger().warn("[LIN_APPROACH] approach_direction is zero; skipping")
                return
            direction = direction / d_norm
            dist_along_dir = max(0.0, float(np.dot(target_np - ee_np, direction)))
        else:
            diff = target_np - ee_np
            dist_along_dir = float(np.linalg.norm(diff))
            if dist_along_dir < 1e-6:
                self.get_logger().warn("[LIN_APPROACH] already at grasp target; skipping")
                return
            direction = diff / dist_along_dir

        planned_distance = min(max(0.0, self.lin_approach_distance_m), dist_along_dir)
        if planned_distance < 1e-4:
            self.get_logger().info("[LIN_APPROACH] planned distance too small; skipping")
            return

        step_m = max(0.005, min(planned_distance, self.lin_approach_step_m))
        threshold = self.lin_approach_obstacle_distance_m
        traveled = 0.0
        executed_waypoints: List[List[float]] = [ee_np.tolist()]

        self.get_logger().info(
            f"[LIN_APPROACH] moving toward grasp by up to {planned_distance:.3f}m "
            f"in {step_m:.3f}m segments"
        )

        while traveled + 1e-6 < planned_distance and rclpy.ok():
            if self.variant != "commit_only":
                if self._lin_approach_is_blocked(threshold):
                    if self.variant == "feas_only":
                        self.get_logger().error(
                            f"[LIN_APPROACH] obstacle within {threshold / INCH_TO_M:.1f}in "
                            f"(feas_only): aborting task"
                        )
                        self._task_result = (
                            False, "feas_only abort: obstacle during lin_approach"
                        )
                        raise _ObstacleAbort()

                    if self.variant == "hold_commit":
                        if not self._hold_commit_paused_once:
                            self.get_logger().warn(
                                f"[LIN_APPROACH] obstacle within {threshold / INCH_TO_M:.1f}in "
                                f"(hold_commit): pausing "
                                f"{self.hold_commit_obstacle_pause_sec:.1f}s then retrying"
                            )
                            time.sleep(self.hold_commit_obstacle_pause_sec)
                            self._hold_commit_paused_once = True
                        else:
                            self.get_logger().warn(
                                f"[LIN_APPROACH] obstacle within {threshold / INCH_TO_M:.1f}in "
                                f"(hold_commit): already paused once, continuing"
                            )
                    elif self.variant == "hac":
                        self.get_logger().warn(
                            f"[LIN_APPROACH] obstacle within {threshold / INCH_TO_M:.1f}in "
                            f"(hac): waiting to clear"
                        )
                        while self._lin_approach_is_blocked(threshold) and rclpy.ok():
                            time.sleep(0.10)
                        self.get_logger().info("[LIN_APPROACH] obstacle cleared; continuing")

            segment = min(step_m, planned_distance - traveled)
            next_traveled = traveled + segment
            waypoint = (ee_np + (next_traveled * direction)).tolist()
            ok = self._move_pose(
                waypoint,
                list(ee_quat),
                phase=f"lin_approach_step_{len(executed_waypoints)}",
                cartesian=True,
            )
            if not ok:
                self.get_logger().warn(
                    f"[LIN_APPROACH] stopped at {traveled:.3f}m/{planned_distance:.3f}m; "
                    f"reversing along executed path"
                )
                self._reverse_lin_approach(executed_waypoints, list(ee_quat))
                return

            traveled = next_traveled
            executed_waypoints.append(waypoint)

        self.get_logger().info(
            f"[LIN_APPROACH] completed forward travel {traveled:.3f}m/{planned_distance:.3f}m"
        )

    # ────────────────────── TASK LOGIC ──────────────────────

    def _run_task(self):
        try:
            self.get_logger().info("[TASK] starting clock pick task")
            scan_poses = [list(map(float, p)) for p in self.scan_poses_raw]

            # 1. WAIT_POSE — brief wait for pose node to come online
            self._set_state("WAIT_POSE")
            t0 = time.time()
            while (time.time() - t0) < 10.0 and rclpy.ok():
                with self._pose_lock:
                    have_pose = self._latest_clock_pose is not None
                if have_pose:
                    break
                time.sleep(0.1)
            # Don't fail — detection may arrive during scan

            # 2. OPEN_GRIPPER
            self._set_state("OPEN_GRIPPER")
            self._mark_started()
            if not self._gripper_ready:
                self.get_logger().warn("[GRIPPER] server not ready; continuing")
            else:
                self.get_logger().info(f"[GRIPPER] opening to {self.gripper_open:.3f}")
                open_sent = self.gripper.command(self.gripper_open)
                if not open_sent:
                    self.get_logger().error("[GRIPPER] open goal rejected")
                else:
                    open_ok = self.gripper.wait(5.0)
                    if open_ok:
                        self.get_logger().info("[GRIPPER] open completed")
                    else:
                        self.get_logger().error("[GRIPPER] open timed out")

            clock_pose = None
            if self.enable_prescan:
                self._set_state("SCAN")
                max_cycles = self.max_scan_cycles
                for cycle in range(max_cycles):
                    for i, joints in enumerate(scan_poses):
                        if self._shutdown_requested or not rclpy.ok():
                            self._task_result = (False, "shutdown")
                            return

                        self.get_logger().info(
                            f"[SCAN] cycle {cycle+1}/{max_cycles} "
                            f"pose {i+1}/{len(scan_poses)}")
                        ok = self._move_joints(joints, phase=f"scan_pose_{cycle}_{i}")
                        if not ok:
                            self.get_logger().warn("[SCAN] move_joints failed; continuing")
                            time.sleep(self.scan_extra_pause_s)
                            continue

                        reached_mono = time.monotonic()
                        while rclpy.ok():
                            if self._shutdown_requested:
                                self._task_result = (False, "shutdown")
                                return
                            clock_pose = self._stable_pose(self.prescan_dwell_s, self.pose_timeout_s)
                            if clock_pose is not None:
                                self._clock_found_scan_idx = i
                                self.get_logger().info(
                                    f"[SCAN] stable clock detected at "
                                    f"({clock_pose[0]:.3f},{clock_pose[1]:.3f},"
                                    f"{clock_pose[2]:.3f},y={clock_pose[3]:.2f}) "
                                    f"from pose {i+1}/{len(scan_poses)}"
                                )
                                self._lock_clock_pose(clock_pose, "scan")
                                break
                            elapsed = time.monotonic() - reached_mono
                            if elapsed >= (self.prescan_pose_settle_s + self.scan_extra_pause_s):
                                break
                            time.sleep(0.05)

                        if clock_pose is not None:
                            break
                    if clock_pose is not None:
                        break
            else:
                clock_pose = self._get_pose_if_fresh(self.pose_timeout_s)

            if clock_pose is None:
                self._task_result = (False, "Clock not detected after all scan cycles")
                return

            self.t_detect_s = self._trial_t()

            # 4. Compute grasp target (end-of-clock corner point)
            cx, cy, cz, yaw_raw = clock_pose
            yaw = float(yaw_raw) + self.yaw_offset_rad

            local_x = (CLOCK_LENGTH_M / 2.0) - float(self.grasp_end_inset_m)
            local_y = 0.0
            local_z = CLOCK_HEIGHT_M / 2.0

            cos_y = math.cos(yaw)
            sin_y = math.sin(yaw)
            offset_x = local_x * cos_y - local_y * sin_y
            offset_y = local_x * sin_y + local_y * cos_y

            grasp_target = [
                float(cx) - offset_x,
                float(cy) - offset_y,
                float(cz) + local_z,
            ]
            if grasp_target[2] < self.min_grasp_z_m:
                grasp_target[2] = self.min_grasp_z_m

            # 4a. Compute grasp orientation and approach geometry (needed for LIN_APPROACH direction)
            self._active_object_pose = (float(cx), float(cy), float(cz))

            orient = self._compute_clock_relative_grasp_quat(grasp_target, yaw)
            if orient is None:
                self._task_result = (False, "Could not compute clock-relative grasp orientation")
                return
            grasp_quat, x_tool, y_tool, z_tool = orient
            quat_source = "clock_relative"

            local_x_backoff_m = self.grasp_local_x_backoff_m
            local_z_backoff_m = self.grasp_local_z_backoff_m
            x_axis = np.array(x_tool, dtype=float)
            z_axis = np.array(z_tool, dtype=float)
            grasp_np = np.array(grasp_target, dtype=float)
            staging_np = grasp_np + (local_x_backoff_m * x_axis) - (local_z_backoff_m * z_axis)
            intermediate_np = grasp_np - (local_z_backoff_m * z_axis)
            staging = staging_np.tolist()
            intermediate = intermediate_np.tolist()

            # 4b. LIN approach along the actual approach axis (z_tool)
            self.get_logger().info(
                f"[DETECT] clock at "
                f"({clock_pose[0]:.3f},{clock_pose[1]:.3f},{clock_pose[2]:.3f})"
            )
            self._hold_commit_paused_once = False
            self._set_state("LIN_APPROACH")
            try:
                self._execute_lin_approach(grasp_target, approach_direction=z_tool)
            except _ObstacleAbort:
                return

            self.get_logger().info("[LIN_APPROACH] returning to scan position")
            scan_return_ok = self._move_joints(
                scan_poses[self._clock_found_scan_idx],
                phase="return_to_scan_after_lin",
            )
            if not scan_return_ok:
                self.get_logger().warn(
                    "[LIN_APPROACH] return-to-scan move failed; continuing with grasp sequence"
                )

            # 5. MOVE_GRASP
            self._hold_commit_paused_once = False
            self._set_state("MOVE_GRASP")
            self.get_logger().info(
                f"[MOVE_GRASP] starting grasp sequence for clock at "
                f"({clock_pose[0]:.3f},{clock_pose[1]:.3f},{clock_pose[2]:.3f})"
            )

            self.get_logger().info(
                f"[MOVE] grasp_target="
                f"({grasp_target[0]:.3f},{grasp_target[1]:.3f},{grasp_target[2]:.3f}) "
                f"yaw={yaw:+.3f} grasp_quat=({grasp_quat[0]:+.3f},{grasp_quat[1]:+.3f},"
                f"{grasp_quat[2]:+.3f},{grasp_quat[3]:+.3f}) "
                f"quat_source={quat_source} "
                f"x_tool=({x_tool[0]:+.3f},{x_tool[1]:+.3f},{x_tool[2]:+.3f}) "
                f"y_tool=({y_tool[0]:+.3f},{y_tool[1]:+.3f},{y_tool[2]:+.3f}) "
                f"z_tool=({z_tool[0]:+.3f},{z_tool[1]:+.3f},{z_tool[2]:+.3f}) "
                f"local_backoff_in=({local_x_backoff_m / INCH_TO_M:.1f}x,"
                f"{local_z_backoff_m / INCH_TO_M:.1f}z) "
                f"staging=({staging[0]:.3f},{staging[1]:.3f},{staging[2]:.3f}) "
                f"intermediate=({intermediate[0]:.3f},{intermediate[1]:.3f},{intermediate[2]:.3f})"
            )
            self._publish_target_pose(self.ee_target_pose_pub, grasp_target, grasp_quat)
            self._publish_target_pose(self.ee_staging_pose_pub, staging, grasp_quat)
            self._publish_target_pose(self.ee_intermediate_pose_pub, intermediate, grasp_quat)

            grasp_reached = False
            if self._obstacle_gate_enabled:
                self._set_obstacle_gate(True, "move_grasp_start")
            try:
                # Phase 1: joint-space to staging (outside shelf, clear of obstacles)
                self.get_logger().info("[MOVE] Phase 1: MoveIt joint-space to staging")
                try:
                    staging_ok = self._obstacle_aware_move(
                        staging, grasp_quat, "staging", cartesian=False)
                except _ObstacleAbort:
                    return
                if not staging_ok:
                    self._task_result = (False, "Could not reach staging pose")
                    return

                # Disable obstacle gate before cartesian moves (clock enters view)
                if self._obstacle_gate_active:
                    self._set_obstacle_gate(False, "staging_reached_clock_in_path")

                self.last_grasp_quat = list(grasp_quat)

                # Phase 2: cartesian move -10in in EE local X.
                self.get_logger().info("[MOVE] Phase 2: Cartesian local -X to intermediate")
                try:
                    x_ok = self._obstacle_aware_move(
                        intermediate, grasp_quat, "x_slide", cartesian=True)
                except _ObstacleAbort:
                    return
                if not x_ok:
                    self._task_result = (False, "Could not reach intermediate pose (X-slide)")
                    return

                # Phase 3: cartesian move +5in in EE local Z to the grasp point.
                self.get_logger().info("[MOVE] Phase 3: Cartesian local +Z to grasp")
                try:
                    lin_ok = self._obstacle_aware_move(
                        grasp_target, grasp_quat, "z_push", cartesian=True)
                except _ObstacleAbort:
                    return
                grasp_reached = bool(lin_ok)
            finally:
                self._active_object_pose = None
                if self._obstacle_gate_active:
                    self._set_obstacle_gate(False, "move_grasp_end")

            if not grasp_reached:
                self._task_result = (False, "Could not reach grasp pose")
                return

            # 6. GRASP — close gripper
            self._set_state("GRASP")
            self.t_grasp_s = self._trial_t()
            if self._gripper_ready:
                self.get_logger().info(f"[GRIPPER] closing to {self.gripper_closed:.3f}")
                close_sent = self.gripper.command(self.gripper_closed)
                if not close_sent:
                    self.get_logger().error("[GRIPPER] close goal rejected")
                else:
                    close_ok = self.gripper.wait(5.0)
                    if close_ok:
                        self.get_logger().info("[GRIPPER] close completed")
                    else:
                        self.get_logger().error("[GRIPPER] close timed out")
            else:
                self.get_logger().warn("[GRIPPER] not ready; skipping close")

            # 7. RELEASE — open gripper (no lift, no place)
            self._set_state("RELEASE")
            self.t_release_s = self._trial_t()
            if self._gripper_ready:
                self.get_logger().info(f"[GRIPPER] releasing to {self.gripper_open:.3f}")
                release_sent = self.gripper.command(self.gripper_open)
                if not release_sent:
                    self.get_logger().error("[GRIPPER] release goal rejected")
                else:
                    release_ok = self.gripper.wait(5.0)
                    if release_ok:
                        self.get_logger().info("[GRIPPER] release completed")
                    else:
                        self.get_logger().error("[GRIPPER] release timed out")

            # 8. RETREAT — reverse the approach path exactly
            self._set_state("RETREAT")

            # Reverse Phase 3 first: local -Z, grasp -> intermediate.
            self.get_logger().info("[RETREAT] Phase 1: Cartesian local -Z back to intermediate")
            ret_z_ok = self._move_pose(
                intermediate, grasp_quat, "reverse_z", cartesian=True)
            if not ret_z_ok:
                self.get_logger().warn("[RETREAT] reverse_z failed; trying joint-space fallback")
                self._move_pose(intermediate, grasp_quat, "reverse_z_jt", cartesian=False)

            # Reverse Phase 2 second: local +X, intermediate -> staging.
            self.get_logger().info("[RETREAT] Phase 2: Cartesian local +X back to staging")
            ret_x_ok = self._move_pose(
                staging, grasp_quat, "reverse_x", cartesian=True)
            if not ret_x_ok:
                self.get_logger().warn("[RETREAT] reverse_x failed; trying joint-space fallback")
                self._move_pose(staging, grasp_quat, "reverse_x_jt", cartesian=False)

            # Step 3: joint-space reverse of the scan poses (staging → scan_poses[0])
            found_idx = self._clock_found_scan_idx
            reverse_poses = list(reversed(scan_poses[: found_idx + 1]))
            for step, joints in enumerate(reverse_poses):
                pose_idx = found_idx - step
                self.get_logger().info(
                    f"[RETREAT] scan pose {pose_idx}/{len(scan_poses) - 1}")
                ok = self._move_joints(joints, phase=f"retreat_scan_{pose_idx}")
                if not ok:
                    self.get_logger().warn(
                        f"[RETREAT] scan pose {pose_idx} failed; continuing")

            # 9. DONE
            self._set_state("DONE")
            self._task_result = (True, "Success")

        except Exception as e:
            self.get_logger().error(f"[TASK] exception: {e}")
            self._task_result = (False, f"exception: {e}")


# ============================
# Entry point
# ============================

def main():
    rclpy.init()
    node = ClockPickSupervisorMetrics()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
