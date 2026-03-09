#!/usr/bin/env python3
"""
bottle_pick_supervisor_metrics.py

Bottle pick task supervisor with scan-point cycling and YOLO detection.

Flow:
1) Set up collision scene (table_shelf, main_table_shelf, wheelchair).
2) Open gripper, then cycle through scan joint configurations.
3) Pause ~2 seconds at each scan point while YOLO runs in background.
4) When bottle is detected, move to standoff pose (joint-space), then
   cartesian LIN approach in +X to grasp pose.
5) Close gripper, move to place joints, release.
6) Record task metrics CSVs.

Key design points:
- Grasp approach uses a two-phase strategy: joint-space move to a
  standoff pose (pulled back in base -X), then a short cartesian LIN
  move in +X to close the final gap (default 20 cm).
- Obstacle gate is active during both standoff and LIN phases, with
  variant-specific handling (commit_only, feas_only, hold_commit, hac).
- Scan is a multi-point loop (like the plant supervisor) instead of a
  single fixed pose.
"""

import csv
import math
import os
import threading
import time
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import message_filters
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import (
    AttachedCollisionObject,
    CollisionObject,
    PlanningScene,
)
from moveit_msgs.srv import ApplyPlanningScene
from pymoveit2 import MoveIt2, MoveIt2State
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import String
from tf2_geometry_msgs import do_transform_pose_stamped
from tf2_ros import Buffer, TransformListener
from ultralytics import YOLO
from visualization_msgs.msg import Marker

from action_msgs.msg import GoalStatus
from control_msgs.action import GripperCommand


# ─────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────

INCH_TO_M = 0.0254
BASE_FRAME = "base_link"
MOVEIT_PKG = "kinova_gen3_6dof_robotiq_2f_85_moveit_config"
GRIPPER_ACTION = "/robotiq_gripper_controller/gripper_cmd"
LEFT_FINGER_TIP_FRAME = "robotiq_85_left_finger_tip_link"
RIGHT_FINGER_TIP_FRAME = "robotiq_85_right_finger_tip_link"
BOTTLE_RADIUS = 0.03175
BOTTLE_HEIGHT = 0.14
SUPPORTED_VARIANTS = {"commit_only", "feas_only", "hold_commit", "hac"}
OBSTACLE_STOP_DISTANCE_M = 3.0 * INCH_TO_M
OBSTACLE_RESUME_DISTANCE_M = 1.0 * INCH_TO_M

ARM_JOINTS = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
]

# ── Scan poses (cycle through these looking for the bottle) ──
SCAN_POSES: List[List[float]] = [
    [-0.1728641884603892, 1.2131819956129855, -2.4910873278303844,
     -0.1499833736388645, 2.029023152905293, 1.5883517515426442],
    [1.5701, 0.7471, 1.3472, 0.1524, 1.2910, -1.6665],
    [1.2839, 0.8440, 1.3817, 0.2653, 1.4562, -1.6168],
    [0.6427, 0.9756, 1.4850, 0.2295, 1.2395, -1.6416],
    [0.5322, 0.8251, 1.2608, 0.1999, 1.6878, -1.6803],
]

DEFAULT_PLACE_JOINTS = [
    1.4810134432811641,
    -0.13624146241405555,
    2.2040947024351945,
    -3.0304498865530807,
    0.8794221975261591,
    1.4390594682066553,
]






# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

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


def _read_pred_flaps(pred_csv: str) -> int:
    if pred_csv is None or str(pred_csv).strip() == "":
        return 0
    pred_csv = str(pred_csv)
    if not os.path.exists(pred_csv):
        raise FileNotFoundError(f"pred_csv not found: {pred_csv}")
    with open(pred_csv, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0
    cs = [str(r.get("commit_state", "")).strip().upper() for r in rows]
    cs = [x for x in cs if x in ("HOLD", "COMMIT")]
    return int(_count_flaps(cs))


def infer_srdf(pkg: str) -> Tuple[str, str, str]:
    share = Path(get_package_share_directory(pkg))
    srdf = next(share.joinpath("config").glob("*.srdf"))
    root = ET.parse(srdf).getroot()
    for g in root.findall("group"):
        chain = g.find("chain")
        if chain is not None:
            return g.get("name"), chain.get("base_link"), chain.get("tip_link")
    raise RuntimeError("No chain group found in SRDF")


def apply_scene(node: "BottlePickSupervisorMetrics", scene: PlanningScene,
                timeout_s: float = 3.0) -> bool:
    req = ApplyPlanningScene.Request()
    req.scene = scene
    future = node.scene_client.call_async(req)
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


# ─────────────────────────────────────────────────────────
# Gripper helper
# ─────────────────────────────────────────────────────────

class Gripper:
    def __init__(self, node: Node, callback_group=None):
        self.node = node
        self.client = ActionClient(
            node, GripperCommand, GRIPPER_ACTION, callback_group=callback_group)
        self.node.get_logger().info("Waiting for gripper action server...")
        self.client.wait_for_server()
        self.node.get_logger().info("Gripper action server ready")

    def command(self, position: float, timeout_s: float = 5.0) -> bool:
        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = 50.0
        future = self.client.send_goal_async(goal)

        t0 = time.time()
        while rclpy.ok() and not future.done():
            if (time.time() - t0) > timeout_s:
                self.node.get_logger().error("Gripper send_goal timeout")
                return False
            time.sleep(0.01)

        gh = future.result()
        if gh is None or not gh.accepted:
            self.node.get_logger().error("Gripper goal rejected")
            return False

        result_future = gh.get_result_async()
        t1 = time.time()
        while rclpy.ok() and not result_future.done():
            if (time.time() - t1) > timeout_s:
                self.node.get_logger().error("Gripper result timeout")
                return False
            time.sleep(0.01)

        try:
            result = result_future.result()
            status = getattr(result, "status", GoalStatus.STATUS_UNKNOWN)
            return int(status) == GoalStatus.STATUS_SUCCEEDED
        except Exception:
            return False


# ─────────────────────────────────────────────────────────
# Main node
# ─────────────────────────────────────────────────────────

class BottlePickSupervisorMetrics(Node):
    def __init__(self):
        super().__init__("bottle_pick_supervisor_metrics")

        self.cbg = ReentrantCallbackGroup()
        self.scene_cbg = ReentrantCallbackGroup()

        # ── Parameters: logging / metrics ──
        self.declare_parameter("trial_id", 0)
        self.declare_parameter("log_dir", "/tmp/bottle_pick_logs")
        self.declare_parameter("global_summary_csv", "/tmp/bottle_pick_metrics_summary.csv")
        self.declare_parameter("pred_csv", "")
        self.declare_parameter("max_runtime_s", 180.0)
        self.declare_parameter("stop_after_success", True)
        self.declare_parameter("stop_after_failure", True)

        self.trial_id = int(self.get_parameter("trial_id").value)
        self.log_dir = str(self.get_parameter("log_dir").value)
        self.global_summary_csv = str(self.get_parameter("global_summary_csv").value)
        self.pred_csv = str(self.get_parameter("pred_csv").value)
        self.max_runtime_s = float(self.get_parameter("max_runtime_s").value)
        self.stop_after_success = bool(self.get_parameter("stop_after_success").value)
        self.stop_after_failure = bool(self.get_parameter("stop_after_failure").value)

        # ── Parameters: motion ──
        self.declare_parameter("place_joints", DEFAULT_PLACE_JOINTS)
        self.declare_parameter("moveit_max_vel", 0.15)
        self.declare_parameter("moveit_max_acc", 0.15)
        self.declare_parameter("planning_pipeline", "ompl")
        self.declare_parameter("planner_id", "")
        self.declare_parameter("planning_time_s", 5.0)
        self.declare_parameter("planning_attempts", 10)
        self.declare_parameter("place_forward_distance_m", 0.2)
        self.declare_parameter("place_back_distance_m", 0.3)
        self.declare_parameter("place_forward_cartesian", True)
        self.declare_parameter("variant", "commit_only")
        self.declare_parameter("hold_commit_obstacle_pause_sec", 5.0)

        self.declare_parameter("gripper_open", 0.0)
        self.declare_parameter("gripper_closed", 0.4)

        self.declare_parameter("grasp_offset_x_m", -0.07)
        self.declare_parameter("grasp_offset_y_m", 0.0)
        self.declare_parameter("grasp_offset_z_m", -0.0027)
        self.declare_parameter("min_grasp_z_m", 0.03)
        self.declare_parameter("grasp_close_enough_m", 0.02)

        self.declare_parameter("scan_dwell_s", 2.0)
        self.declare_parameter("max_scan_cycles", 5)

        self.declare_parameter("grasp_lin_approach_m", 0.07)
        self.declare_parameter("grasp_step_m", 0.02)

        self.place_joints_raw = [float(x) for x in list(self.get_parameter("place_joints").value)]
        self.moveit_max_vel = float(self.get_parameter("moveit_max_vel").value)
        self.moveit_max_acc = float(self.get_parameter("moveit_max_acc").value)
        self.planning_pipeline = str(self.get_parameter("planning_pipeline").value)
        self.planner_id = str(self.get_parameter("planner_id").value)
        self.planning_time_s = float(self.get_parameter("planning_time_s").value)
        self.planning_attempts = int(self.get_parameter("planning_attempts").value)
        self.place_forward_distance_m = float(self.get_parameter("place_forward_distance_m").value)
        self.place_back_distance_m = float(self.get_parameter("place_back_distance_m").value)
        self.place_forward_cartesian = bool(self.get_parameter("place_forward_cartesian").value)
        self.variant = str(self.get_parameter("variant").value).strip().lower()
        if self.variant not in SUPPORTED_VARIANTS:
            self.get_logger().warn(
                f"Unsupported variant '{self.variant}', falling back to 'commit_only'")
            self.variant = "commit_only"
        self.hold_commit_obstacle_pause_sec = float(
            self.get_parameter("hold_commit_obstacle_pause_sec").value)
        self.gripper_open = float(self.get_parameter("gripper_open").value)
        self.gripper_closed = float(self.get_parameter("gripper_closed").value)

        self.grasp_offset_x_m = float(self.get_parameter("grasp_offset_x_m").value)
        self.grasp_offset_y_m = float(self.get_parameter("grasp_offset_y_m").value)
        self.grasp_offset_z_m = float(self.get_parameter("grasp_offset_z_m").value)
        self.min_grasp_z_m = float(self.get_parameter("min_grasp_z_m").value)
        self.grasp_close_enough_m = float(self.get_parameter("grasp_close_enough_m").value)

        self.scan_dwell_s = float(self.get_parameter("scan_dwell_s").value)
        self.max_scan_cycles = int(self.get_parameter("max_scan_cycles").value)

        self.grasp_lin_approach_m = float(self.get_parameter("grasp_lin_approach_m").value)
        self.grasp_step_m = float(self.get_parameter("grasp_step_m").value)

        # Trim place joints to arm length
        self.place_joints = self._arm_joint_target(self.place_joints_raw, "place_joints")

        # ── Parameters: perception ──
        self.declare_parameter("camera_rgb_topic", "/camera/color/image_raw")
        self.declare_parameter("camera_depth_topic", "/camera/depth_registered/image_rect")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("yolo_model", "yolov8n-seg.pt")
        self.declare_parameter("yolo_period_s", 0.2)
        self.declare_parameter("confidence_threshold", 0.5)
        self.declare_parameter("pose_timeout_s", 2.0)
        self.declare_parameter("detection_timeout_s", 8.0)
        self.declare_parameter("min_depth_m", 0.15)
        self.declare_parameter("max_depth_m", 1.5)
        self.declare_parameter("smoothing_alpha", 0.3)
        self.declare_parameter("workspace_x_min_m", -1.0)
        self.declare_parameter("workspace_x_max_m", 1.0)
        self.declare_parameter("workspace_y_min_m", -1.0)
        self.declare_parameter("workspace_y_max_m", 1.0)
        self.declare_parameter("workspace_z_min_m", -0.1)
        self.declare_parameter("workspace_z_max_m", 1.5)
        self.declare_parameter("obstacle_monitor_enabled", True)
        self.declare_parameter("obstacle_roi_fraction", 0.30)
        self.declare_parameter("obstacle_min_valid_pixels", 80)
        self.declare_parameter("obstacle_percentile", 5.0)
        self.declare_parameter("obstacle_median_window", 5)
        self.declare_parameter("obstacle_stop_confirm_frames", 2)
        self.declare_parameter("obstacle_resume_confirm_frames", 3)
        self.declare_parameter("obstacle_depth_timeout_sec", 1.0)
        self.declare_parameter("obstacle_fail_safe_on_no_depth", True)
        self.declare_parameter("obstacle_stop_distance_m", OBSTACLE_STOP_DISTANCE_M)
        self.declare_parameter("obstacle_resume_distance_m", OBSTACLE_RESUME_DISTANCE_M)
        self.declare_parameter("obstacle_ignore_bottle_within_m", 3.0 * INCH_TO_M)

        self.camera_rgb_topic = str(self.get_parameter("camera_rgb_topic").value)
        self.camera_depth_topic = str(self.get_parameter("camera_depth_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self.yolo_model_path = str(self.get_parameter("yolo_model").value)
        self.yolo_period_s = float(self.get_parameter("yolo_period_s").value)
        self.confidence_threshold = float(self.get_parameter("confidence_threshold").value)
        self.pose_timeout_s = float(self.get_parameter("pose_timeout_s").value)
        self.detection_timeout_s = float(self.get_parameter("detection_timeout_s").value)
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)
        self.smoothing_alpha = float(self.get_parameter("smoothing_alpha").value)
        self.workspace_bounds = {
            "x_min": float(self.get_parameter("workspace_x_min_m").value),
            "x_max": float(self.get_parameter("workspace_x_max_m").value),
            "y_min": float(self.get_parameter("workspace_y_min_m").value),
            "y_max": float(self.get_parameter("workspace_y_max_m").value),
            "z_min": float(self.get_parameter("workspace_z_min_m").value),
            "z_max": float(self.get_parameter("workspace_z_max_m").value),
        }
        self.obstacle_monitor_enabled = bool(
            self.get_parameter("obstacle_monitor_enabled").value)
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
        self.obstacle_stop_distance_m = float(
            self.get_parameter("obstacle_stop_distance_m").value)
        self.obstacle_resume_distance_m = float(
            self.get_parameter("obstacle_resume_distance_m").value)
        self.obstacle_ignore_bottle_within_m = float(
            self.get_parameter("obstacle_ignore_bottle_within_m").value)

        # ── Parameters: collision objects ──
        self.declare_parameter("collision_enabled", True)
        self.declare_parameter("table_point", [0.86, 0., -0.31645])
        self.declare_parameter("table_yaw", 0.0)
        self.declare_parameter("main_table_point", [-0.5156, 0.6, 0.18645])
        self.declare_parameter("main_table_yaw", math.pi / 2.0)
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
        self.collision_enabled = bool(self.get_parameter("collision_enabled").value)

        # ── Metrics state ──
        self.trial_t0 = time.monotonic()
        os.makedirs(self.log_dir, exist_ok=True)
        self.state_path = os.path.join(self.log_dir, "state_stream.csv")
        self.attempts_path = os.path.join(self.log_dir, "attempts.csv")
        self.metrics_trial_path = os.path.join(self.log_dir, "metrics_trial.csv")

        self._init_csv(self.state_path, ["t_s", "trial_id", "state"])
        self._init_csv(self.attempts_path,
                       ["trial_id", "started_flag", "t_start_s", "t_end_s", "outcome", "detail"])
        self._init_csv(self.metrics_trial_path, [
            "trial_id", "variant", "started_flag", "outcome",
            "t_start_s", "t_detect_s", "t_grasp_s", "t_release_s", "t_success_s",
            "time_to_detect_s", "time_to_grasp_s", "time_to_release_s", "time_to_success_s",
            "flaps",
        ])

        self._global_header = [
            "trial_id", "variant", "log_dir", "pred_csv", "started_flag", "outcome",
            "t_start_s", "t_detect_s", "t_grasp_s", "t_release_s", "t_success_s",
            "time_to_detect_s", "time_to_grasp_s", "time_to_release_s", "time_to_success_s",
            "flaps",
        ]
        self._init_global_summary(self.global_summary_csv, self._global_header)

        self.started_flag = 0
        self.t_start_s = float("nan")
        self.t_detect_s = float("nan")
        self.t_grasp_s = float("nan")
        self.t_release_s = float("nan")
        self.t_success_s = float("nan")
        self.outcome = "timeout"
        self.flaps = 0
        self._final_saved = False

        # ── Task state ──
        self.state = "INIT"
        self._finished = False
        self._shutdown_requested = False
        self._task_result: Optional[Tuple[bool, str]] = None
        self._scene_lock = threading.Lock()
        self._scene_ready = (not self.collision_enabled)
        self._scene_retry_period_s = 2.0
        self._scene_next_try_s = 0.0
        self._task_thread: Optional[threading.Thread] = None
        self._last_motion_abort_reason = ""

        # ── Perception state ──
        self.bridge = CvBridge()
        self.vision_lock = threading.Lock()
        self.detection_lock = threading.Lock()
        self._obstacle_lock = threading.Lock()

        self.camera_info = None
        self.fx = self.fy = self.cx = self.cy = None

        self._latest_rgb_msg = None
        self._latest_depth_msg = None
        self._new_pair_event = threading.Event()
        self._stop_event = threading.Event()
        self._last_yolo_time = 0.0
        self._last_reject_log_t = 0.0

        self.tracked_pose: Optional[Tuple[float, float, float]] = None
        self.pose_timestamp: Optional[float] = None
        self.last_grasp_quat: Optional[List[float]] = None
        self._active_bottle_pose: Optional[Tuple[float, float, float]] = None

        # Obstacle gate only applies during MOVE_GRASP in variants that use it.
        self._obstacle_gate_enabled = (
            self.obstacle_monitor_enabled
            and self.variant in ("feas_only", "hold_commit", "hac")
        )
        self._obstacle_gate_active = False
        self._obstacle_last_depth_mono = time.monotonic()
        self._obstacle_closest_hist = deque(maxlen=max(1, self.obstacle_median_window))
        self._obstacle_closest_dist = float("inf")
        self._obstacle_blocked = False
        self._obstacle_stop_hits = 0
        self._obstacle_clear_hits = 0
        self._obstacle_timeout_active = False
        self._obstacle_min_depth_m = float("inf")
        self._obstacle_ignore_last_log_mono = 0.0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # COCO class 39 = bottle
        self.target_classes = {39}

        # ── Publishers ──
        self.debug_image_pub = self.create_publisher(
            Image, "/bottle_detection/debug_image", 10)
        self.bottle_marker_pub = self.create_publisher(
            Marker, "/bottle_pick/bottle_marker", 10)
        self.state_pub = self.create_publisher(
            String, "/bottle_pick/state", 10)

        # ── Planning-scene service ──
        self.scene_client = self.create_client(
            ApplyPlanningScene, "/apply_planning_scene",
            callback_group=self.scene_cbg)
        while not self.scene_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /apply_planning_scene...")

        # ── YOLO model ──
        self.get_logger().info(f"Loading YOLO model: {self.yolo_model_path}")
        self.model = YOLO(self.yolo_model_path)
        self.get_logger().info("YOLO model loaded")

        # ── Camera subscriptions ──
        self.camera_info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic,
            self._camera_info_callback, 10,
            callback_group=self.cbg)

        self.rgb_sub = message_filters.Subscriber(self, Image, self.camera_rgb_topic)
        self.depth_sub = message_filters.Subscriber(self, Image, self.camera_depth_topic)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], 10, 0.1)
        self.sync.registerCallback(self._image_callback)

        # ── YOLO background worker ──
        self._worker_thread = threading.Thread(
            target=self._yolo_worker, daemon=True)
        self._worker_thread.start()

        # ── MoveIt2 setup ──
        group, base_link, tip_link = infer_srdf(MOVEIT_PKG)
        self._group_name = group
        self._base_link = base_link
        self._tip_link = tip_link

        self.moveit = MoveIt2(
            node=self,
            joint_names=ARM_JOINTS,
            base_link_name=base_link,
            end_effector_name=tip_link,
            group_name=group,
            use_move_group_action=True,
            callback_group=self.cbg,
        )
        self.moveit.max_velocity = self.moveit_max_vel
        self.moveit.max_acceleration = self.moveit_max_acc
        self.moveit.pipeline_id = self.planning_pipeline
        self.moveit.planner_id = self.planner_id
        self.moveit.allowed_planning_time = self.planning_time_s
        self.moveit.num_planning_attempts = self.planning_attempts

        # ── Gripper ──
        self.gripper = Gripper(self, callback_group=self.cbg)

        if self.collision_enabled:
            self.get_logger().info("Collision scene setup deferred until executor spin")

        # ── Timers ──
        self.create_timer(0.25, self._log_state_stream, callback_group=self.cbg)
        self.create_timer(0.1, self._tick, callback_group=self.cbg)
        self.create_timer(0.25, self._watchdog_timeout, callback_group=self.cbg)

        self.get_logger().info(
            f"BottlePickSupervisorMetrics up  trial={self.trial_id} "
            f"pipeline={self.planning_pipeline or '<default>'} "
            f"planner_id={self.planner_id or '<default>'} "
            f"variant={self.variant} "
            f"planning_time={self.planning_time_s:.1f}s attempts={self.planning_attempts} "
            f"scan_dwell={self.scan_dwell_s:.1f}s max_scan_cycles={self.max_scan_cycles} "
            f"place_forward={self.place_forward_distance_m:.3f}m "
            f"place_back={self.place_back_distance_m:.3f}m "
            f"place_forward_cartesian={int(self.place_forward_cartesian)} "
            f"obstacle_gate_enabled={int(self._obstacle_gate_enabled)} "
            f"obs_ignore_bottle_within={self.obstacle_ignore_bottle_within_m:.3f}m "
            f"grasp_lin_approach={self.grasp_lin_approach_m:.3f}m "
            f"grasp_step={self.grasp_step_m:.3f}m "
            f"grasp_close_enough={self.grasp_close_enough_m:.3f}m "
            f"place_joints={np.round(self.place_joints, 3).tolist()}"
        )
        if self.pred_csv.strip():
            self.get_logger().info(
                f"pred_csv provided ({self.pred_csv}) "
                "for compatibility; bottle supervisor does not use commit timing from CSV.")
        try:
            self.flaps = _read_pred_flaps(self.pred_csv)
            if self.pred_csv.strip():
                self.get_logger().info(
                    f"pred_csv metadata loaded: flaps={self.flaps}")
        except Exception as e:
            self.flaps = 0
            self.get_logger().warn(
                f"Could not derive flaps from pred_csv ({e}); using flaps=0")

    # ─────────────────────────────────────────────────────
    # CSV / metrics helpers
    # ─────────────────────────────────────────────────────

    def _init_csv(self, path: str, header: List[str]) -> None:
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(header)

    def _init_global_summary(self, path: str, header: List[str]) -> None:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(header)

    def _append_csv(self, path: str, row: List):
        try:
            with open(path, "a", newline="") as f:
                csv.writer(f).writerow(row)
        except Exception:
            pass

    def _trial_t(self) -> float:
        return time.monotonic() - self.trial_t0

    def _mark_started(self):
        if self.started_flag == 0:
            self.started_flag = 1
            self.t_start_s = self._trial_t()

    def _write_metrics_files(self):
        t_detect = (self.t_detect_s - self.t_start_s
                    if self.started_flag and math.isfinite(self.t_detect_s)
                    and math.isfinite(self.t_start_s) else float("nan"))
        t_grasp = (self.t_grasp_s - self.t_start_s
                   if self.started_flag and math.isfinite(self.t_grasp_s)
                   and math.isfinite(self.t_start_s) else float("nan"))
        t_release = (self.t_release_s - self.t_start_s
                     if self.started_flag and math.isfinite(self.t_release_s)
                     and math.isfinite(self.t_start_s) else float("nan"))
        t_success = (self.t_success_s - self.t_start_s
                     if self.started_flag and math.isfinite(self.t_success_s)
                     and math.isfinite(self.t_start_s) else float("nan"))

        row = [
            self.trial_id, self.variant, int(self.started_flag), self.outcome,
            self.t_start_s, self.t_detect_s, self.t_grasp_s, self.t_release_s, self.t_success_s,
            t_detect, t_grasp, t_release, t_success, int(self.flaps),
        ]
        self._append_csv(self.metrics_trial_path, row)
        self._append_csv(self.global_summary_csv, [
            self.trial_id, self.variant, self.log_dir, self.pred_csv,
            int(self.started_flag), self.outcome,
            self.t_start_s, self.t_detect_s, self.t_grasp_s, self.t_release_s, self.t_success_s,
            t_detect, t_grasp, t_release, t_success, int(self.flaps),
        ])

    def _finish(self, outcome: str, detail: str = ""):
        if self._final_saved:
            return
        self._final_saved = True
        self.outcome = str(outcome)
        if self.outcome == "success" and not math.isfinite(self.t_success_s):
            self.t_success_s = self._trial_t()
        self._publish_state(f"FINISH_{self.outcome.upper()}")
        try:
            self._clear_bottle_marker()
        except Exception:
            pass

        self._append_csv(
            self.attempts_path,
            [self.trial_id, int(self.started_flag), self.t_start_s,
             self._trial_t(), self.outcome, detail[:240]],
        )
        self._write_metrics_files()
        self.get_logger().info(
            f"[FINISH] outcome={self.outcome} started={self.started_flag} detail={detail}")

    # ─────────────────────────────────────────────────────
    # State / timers / shutdown
    # ─────────────────────────────────────────────────────

    def _publish_state(self, st: str):
        try:
            msg = String()
            msg.data = str(st)
            self.state_pub.publish(msg)
        except Exception:
            pass

    def _set_state(self, st: str):
        st = str(st)
        if self.state != st:
            self.get_logger().info(f"[STATE] {self.state} -> {st}")
        self.state = st
        self._publish_state(st)

    def _log_state_stream(self):
        if self._finished:
            return
        self._append_csv(self.state_path, [self._trial_t(), self.trial_id, self.state])

    def _request_shutdown(self):
        self._shutdown_requested = True

    def _shutdown_now(self):
        self._finished = True
        self._stop_event.set()
        self._new_pair_event.set()
        try:
            if self._worker_thread.is_alive():
                self._worker_thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            self.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass

    def _watchdog_timeout(self):
        if self._finished:
            return
        if self.max_runtime_s > 0.0 and self._trial_t() > self.max_runtime_s:
            self.get_logger().error("Timeout reached")
            self._finish("timeout", "watchdog")
            self._request_shutdown()

    def _tick(self):
        if self._finished:
            return
        if self._shutdown_requested:
            self._shutdown_now()
            return

        # Retry collision scene until ready
        if self.collision_enabled and not self._scene_ready:
            now = time.monotonic()
            if now >= self._scene_next_try_s:
                self._scene_next_try_s = now + self._scene_retry_period_s
                self._scene_ready = self._setup_collision_objects(log=True, blocking=False)
                if not self._scene_ready:
                    self.get_logger().warn("Collision scene apply failed; will retry")
            return

        # Launch main task thread once
        if self._task_thread is None:
            self._task_thread = threading.Thread(target=self._run_task, daemon=True)
            self._task_thread.start()
            return
        if self._task_thread.is_alive():
            return

        # Task finished
        success, msg = self._task_result or (False, "No result")
        if success:
            self._finish("success", msg)
            if self.stop_after_success:
                self._request_shutdown()
        else:
            self._finish("failed", msg)
            if self.stop_after_failure:
                self._request_shutdown()

    # ─────────────────────────────────────────────────────
    # Collision scene (table + main table/shelf + wheelchair)
    # ─────────────────────────────────────────────────────

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

    def _apply_base_to_local(self, base_pose: Pose, local_pose: Pose) -> Pose:
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

    def _build_collision_objects(self) -> Tuple[CollisionObject, CollisionObject, CollisionObject]:
        table_point = list(self.get_parameter("table_point").value)
        table_yaw = float(self.get_parameter("table_yaw").value)
        main_table_point = list(self.get_parameter("main_table_point").value)
        main_table_yaw = float(self.get_parameter("main_table_yaw").value)
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
        side_thickness_m = (
            float(self.get_parameter("small_table_wall_thickness_in").value) * INCH_TO_M
        )
        back_wall_full_height = bool(self.get_parameter("small_table_back_wall_full_height").value)
        back_wall_height_m = float(self.get_parameter("small_table_back_wall_height_in").value) * INCH_TO_M
        side_thickness_m = max(0.005, side_thickness_m)
        side_extension_m = max(0.0, side_height_m - table_height_m)

        base_table_pose = self._make_collision_pose(table_point, table_yaw)
        base_main_table_pose = self._make_collision_pose(main_table_point, main_table_yaw)
        base_wheel_pose = self._make_collision_pose(wheelchair_point, wheelchair_yaw)

        # ── Main table + shelf geometry (identical to original) ──
        full_table_size = [0.762 + 0.0127, 1.8288, 0.2921 + 0.0127]
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

        full_table_prims: List[Tuple[SolidPrimitive, Pose]] = []
        full_table_prims.append((self._make_box(full_table_size), table_abs_pose))

        for z_rel in shelf_heights:
            z = shelf_origin_z + z_rel + shelf_thickness / 2.0
            full_table_prims.append((
                self._make_box([shelf_outer_size[0], shelf_outer_size[1], shelf_thickness]),
                self._make_collision_pose((shelf_origin_x + shelf_outer_size[0] / 2.0, 0.0, z)),
            ))

        full_table_prims.append((
            self._make_box([shelf_outer_size[0], 0.02, shelf_outer_size[2]]),
            self._make_collision_pose((
                shelf_origin_x + shelf_outer_size[0] / 2.0,
                shelf_origin_y - shelf_outer_size[1] / 2.0,
                shelf_origin_z + shelf_outer_size[2] / 2.0)),
        ))
        full_table_prims.append((
            self._make_box([shelf_outer_size[0], 0.02, shelf_outer_size[2]]),
            self._make_collision_pose((
                shelf_origin_x + shelf_outer_size[0] / 2.0,
                shelf_origin_y + shelf_outer_size[1] / 2.0,
                shelf_origin_z + shelf_outer_size[2] / 2.0)),
        ))
        full_table_prims.append((
            self._make_box([0.02, shelf_outer_size[1], shelf_outer_size[2]]),
            self._make_collision_pose((
                shelf_origin_x + shelf_outer_size[0] - 0.01,
                0.0,
                shelf_origin_z + shelf_outer_size[2] / 2.0)),
        ))
        full_table_prims.append((
            self._make_box([0.02, shelf_outer_size[1], bottom_to_shelf1 - half_shelf_thickness]),
            self._make_collision_pose((
                shelf_origin_x, 0.0, shelf_origin_z + bottom_to_shelf1 / 2.0)),
        ))
        middle_to_part = (shelf_outer_size[1] / 2.0) - left_to_part1
        full_table_prims.append((
            self._make_box([shelf_outer_size[0], 0.02, shelf1_to_shelf2]),
            self._make_collision_pose((
                shelf_origin_x + shelf_outer_size[0] / 2.0,
                middle_to_part,
                shelf_origin_z + shelf1_to_shelf2)),
        ))
        full_table_prims.append((
            self._make_box([shelf_outer_size[0], 0.02, shelf1_to_shelf2]),
            self._make_collision_pose((
                shelf_origin_x + shelf_outer_size[0] / 2.0,
                -middle_to_part,
                shelf_origin_z + shelf1_to_shelf2)),
        ))
        full_table_prims.append((
            self._make_box([shelf_outer_size[0], 0.02, shelf2_to_top]),
            self._make_collision_pose((
                shelf_origin_x + shelf_outer_size[0] / 2.0,
                0.0,
                shelf_origin_z + bottom_to_shelf1 + shelf2_to_top - shelf_thickness / 2.0)),
        ))

        # Optional small box obstacle on the left side of the big table shelf.
        if big_left_box_enabled:
            box_length_m = max(0.01, big_left_box_length_m)
            box_width_m = max(0.01, big_left_box_width_m)
            box_height_m = max(0.01, big_left_box_height_m)
            left_shelf_edge_y = shelf_origin_y + (shelf_outer_size[1] / 2.0)
            box_center_x = shelf_origin_x + (shelf_outer_size[0] / 2.0) + big_left_box_x_offset_m
            box_center_y = left_shelf_edge_y + (box_width_m / 2.0) + big_left_box_gap_m
            box_center_z = full_table_size[2] + (box_height_m / 2.0)
            full_table_prims.append((
                self._make_box([box_length_m, box_width_m, box_height_m]),
                self._make_collision_pose((box_center_x, box_center_y, box_center_z), big_left_box_yaw_rad),
            ))

        co_main_table = CollisionObject()
        co_main_table.id = "main_table_shelf"
        co_main_table.header.frame_id = BASE_FRAME
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

        # ── Small table (table_shelf) ──
        co_table = CollisionObject()
        co_table.id = "table_shelf"
        co_table.header.frame_id = BASE_FRAME
        co_table.operation = CollisionObject.ADD

        local = Pose()
        local.orientation.w = 1.0
        co_table.primitives.append(self._make_box([table_depth_m, table_width_m, table_height_m]))
        co_table.primitive_poses.append(self._apply_base_to_local(base_table_pose, local))

        if side_extension_m > 1e-6:
            lip_center_z = (table_height_m / 2.0) + (side_extension_m / 2.0)

            left = Pose()
            left.position.y = (table_width_m / 2.0) - (side_thickness_m / 2.0)
            left.position.z = lip_center_z
            left.orientation.w = 1.0
            co_table.primitives.append(self._make_box([table_depth_m, side_thickness_m, side_extension_m]))
            co_table.primitive_poses.append(self._apply_base_to_local(base_table_pose, left))

            right = Pose()
            right.position.y = -((table_width_m / 2.0) - (side_thickness_m / 2.0))
            right.position.z = lip_center_z
            right.orientation.w = 1.0
            co_table.primitives.append(self._make_box([table_depth_m, side_thickness_m, side_extension_m]))
            co_table.primitive_poses.append(self._apply_base_to_local(base_table_pose, right))

        back_height_m = back_wall_height_m if back_wall_full_height else side_extension_m
        if back_height_m > 1e-6:
            back = Pose()
            back.position.x = (table_depth_m / 2.0) - (side_thickness_m / 2.0)
            back.position.z = back_height_m / 2.0
            back.orientation.w = 1.0
            co_table.primitives.append(self._make_box([side_thickness_m, table_width_m, back_height_m]))
            co_table.primitive_poses.append(self._apply_base_to_local(base_table_pose, back))

        # ── Wheelchair ──
        wall_size = [0.51, 0.5, 0.2413]
        co_wheel = CollisionObject()
        co_wheel.id = "wheelchair"
        co_wheel.header.frame_id = BASE_FRAME
        co_wheel.operation = CollisionObject.ADD

        local = Pose()
        local.orientation.w = 1.0
        world_pose = self._apply_base_to_local(base_wheel_pose, local)
        co_wheel.primitives.append(self._make_box(wall_size))
        co_wheel.primitive_poses.append(world_pose)

        return co_main_table, co_table, co_wheel

    def _setup_collision_objects(self, log: bool = True, blocking: bool = True) -> bool:
        acquired = self._scene_lock.acquire(blocking=blocking)
        if not acquired:
            return False
        try:
            co_main_table, co_table, co_wheel = self._build_collision_objects()
            scene = PlanningScene()
            scene.world.collision_objects.append(co_main_table)
            scene.world.collision_objects.append(co_table)
            scene.world.collision_objects.append(co_wheel)
            scene.is_diff = True
            ok = apply_scene(self, scene)
            if ok and log:
                self.get_logger().info(
                    "Collision objects applied: "
                    f"'main_table_shelf' ({len(co_main_table.primitives)}), "
                    f"'table_shelf' ({len(co_table.primitives)}), "
                    f"'wheelchair' ({len(co_wheel.primitives)})"
                )
            return ok
        finally:
            self._scene_lock.release()

    # ─────────────────────────────────────────────────────
    # Perception (YOLO bottle detection)
    # ─────────────────────────────────────────────────────

    def _camera_info_callback(self, msg: CameraInfo):
        if self.camera_info is None:
            self.camera_info = msg
            self.fx, self.fy = msg.k[0], msg.k[4]
            self.cx, self.cy = msg.k[2], msg.k[5]
            self.get_logger().info("Camera intrinsics loaded")

    def _image_callback(self, rgb_msg: Image, depth_msg: Image):
        if self.camera_info is None:
            return
        self._on_obstacle_depth_frame(depth_msg)
        with self.vision_lock:
            self._latest_rgb_msg = rgb_msg
            self._latest_depth_msg = depth_msg
        self._new_pair_event.set()

    def _pub_debug(self, img, ref_msg):
        try:
            m = self.bridge.cv2_to_imgmsg(img, "bgr8")
            m.header = ref_msg.header
            self.debug_image_pub.publish(m)
        except Exception:
            pass

    def _publish_bottle_marker(self, bottle_xyz: Tuple[float, float, float]):
        marker = Marker()
        marker.header.frame_id = BASE_FRAME
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "bottle_pick"
        marker.id = 1
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = float(bottle_xyz[0])
        marker.pose.position.y = float(bottle_xyz[1])
        marker.pose.position.z = float(bottle_xyz[2])
        marker.pose.orientation.w = 1.0
        marker.scale.x = 2.0 * BOTTLE_RADIUS
        marker.scale.y = 2.0 * BOTTLE_RADIUS
        marker.scale.z = BOTTLE_HEIGHT
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.65
        self.bottle_marker_pub.publish(marker)

    def _clear_bottle_marker(self):
        marker = Marker()
        marker.header.frame_id = BASE_FRAME
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "bottle_pick"
        marker.id = 1
        marker.action = Marker.DELETE
        self.bottle_marker_pub.publish(marker)

    def _in_ws(self, x: float, y: float, z: float) -> bool:
        b = self.workspace_bounds
        return (
            b["x_min"] <= x <= b["x_max"]
            and b["y_min"] <= y <= b["y_max"]
            and b["z_min"] <= z <= b["z_max"]
        )

    def _log_rejected_pose(self, wx: float, wy: float, wz: float, reason: str):
        now = time.time()
        if (now - self._last_reject_log_t) < 1.0:
            return
        self._last_reject_log_t = now
        self.get_logger().warn(
            f"[DETECT] rejected bottle pose ({wx:.3f},{wy:.3f},{wz:.3f}): {reason}"
        )

    def _fmt_depth_m(self, v: float) -> str:
        if math.isinf(v):
            return "inf"
        if math.isnan(v):
            return "nan"
        return f"{v:.3f}m"

    def _obstacle_snapshot(self) -> Tuple[float, bool, float, float]:
        now = time.monotonic()
        with self._obstacle_lock:
            closest = float(self._obstacle_closest_dist)
            blocked = bool(self._obstacle_blocked)
            min_seen = float(self._obstacle_min_depth_m)
            age = now - float(self._obstacle_last_depth_mono)
        return closest, blocked, min_seen, age

    def _set_obstacle_gate(self, active: bool, reason: str) -> None:
        if not self._obstacle_gate_enabled:
            return
        active = bool(active)
        if self._obstacle_gate_active == active:
            return
        self._obstacle_gate_active = active
        closest, blocked, min_seen, age = self._obstacle_snapshot()
        self.get_logger().info(
            f"[OBS] gate={'ON' if active else 'OFF'} reason={reason} "
            f"closest={self._fmt_depth_m(closest)} blocked={int(blocked)} "
            f"min_seen={self._fmt_depth_m(min_seen)} age={age:.2f}s")

    def _on_obstacle_depth_frame(self, msg: Image) -> None:
        if not self._obstacle_gate_enabled:
            return
        try:
            depth_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception:
            return

        depth = np.asarray(depth_raw, dtype=np.float32)
        if depth.size == 0 or depth.ndim < 2:
            return
        if float(np.nanmax(depth)) > 100.0:
            depth = depth * 0.001

        h, w = depth.shape[:2]
        rh = max(1, int(h * self.obstacle_roi_fraction / 2.0))
        rw = max(1, int(w * self.obstacle_roi_fraction / 2.0))
        cy = h // 2
        cx = w // 2
        y0 = max(0, cy - rh)
        y1 = min(h, cy + rh)
        x0 = max(0, cx - rw)
        x1 = min(w, cx + rw)
        roi = depth[y0:y1, x0:x1]
        valid = roi[np.isfinite(roi) & (roi > 0.0)]
        now = time.monotonic()

        with self._obstacle_lock:
            self._obstacle_last_depth_mono = now
            if valid.size >= self.obstacle_min_valid_pixels:
                raw_closest = float(np.percentile(valid, self.obstacle_percentile))
                self._obstacle_closest_hist.append(raw_closest)
                closest = float(np.median(np.asarray(self._obstacle_closest_hist, dtype=float)))
                self._obstacle_closest_dist = closest
                self._obstacle_min_depth_m = min(float(self._obstacle_min_depth_m), closest)

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

                if (not was_blocked and self._obstacle_stop_hits >= self.obstacle_stop_confirm_frames):
                    self._obstacle_blocked = True
                    self._obstacle_stop_hits = 0
                elif (was_blocked and self._obstacle_clear_hits >= self.obstacle_resume_confirm_frames):
                    self._obstacle_blocked = False
                    self._obstacle_clear_hits = 0
            else:
                self._obstacle_closest_dist = float("inf")
                if self.obstacle_fail_safe_on_no_depth:
                    self._obstacle_stop_hits += 1
                    self._obstacle_clear_hits = 0
                    if (not self._obstacle_blocked
                            and self._obstacle_stop_hits >= self.obstacle_stop_confirm_frames):
                        self._obstacle_blocked = True
                        self._obstacle_stop_hits = 0
                else:
                    self._obstacle_clear_hits += 1
                    self._obstacle_stop_hits = 0
                    if (self._obstacle_blocked
                            and self._obstacle_clear_hits >= self.obstacle_resume_confirm_frames):
                        self._obstacle_blocked = False
                        self._obstacle_clear_hits = 0

    def _obstacle_is_blocked(self) -> bool:
        if not self._obstacle_gate_active:
            return False
        if self._should_ignore_obstacle_for_bottle():
            return False
        now = time.monotonic()
        with self._obstacle_lock:
            blocked = bool(self._obstacle_blocked)
            if (self.obstacle_fail_safe_on_no_depth
                    and self.obstacle_depth_timeout_sec > 0.0):
                age = now - float(self._obstacle_last_depth_mono)
                if age > self.obstacle_depth_timeout_sec:
                    blocked = True
                    if not self._obstacle_timeout_active:
                        self._obstacle_timeout_active = True
                        self._obstacle_blocked = True
                elif self._obstacle_timeout_active:
                    self._obstacle_timeout_active = False
            return blocked

    def _yolo_worker(self):
        while rclpy.ok() and not self._stop_event.is_set():
            if not self._new_pair_event.wait(timeout=0.5):
                continue
            self._new_pair_event.clear()

            now = time.time()
            if (now - self._last_yolo_time) < self.yolo_period_s:
                continue
            self._last_yolo_time = now

            with self.vision_lock:
                rgb_msg = self._latest_rgb_msg
                depth_msg = self._latest_depth_msg
            if rgb_msg is None or depth_msg is None:
                continue

            try:
                rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
                depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
            except Exception:
                continue

            try:
                results = self.model(rgb_image, verbose=False)[0]
            except Exception:
                continue

            debug_image = rgb_image.copy()
            if results.masks is None or results.boxes is None or len(results.boxes) == 0:
                self._pub_debug(debug_image, rgb_msg)
                continue

            try:
                xyxy = results.boxes.xyxy.cpu().numpy()
                cls = results.boxes.cls.cpu().numpy().astype(int)
                conf = results.boxes.conf.cpu().numpy()
                masks = results.masks.data.cpu().numpy()
            except Exception:
                self._pub_debug(debug_image, rgb_msg)
                continue

            best_i = -1
            best_conf = 0.0
            for i in range(len(conf)):
                if cls[i] in self.target_classes and conf[i] >= self.confidence_threshold and conf[i] > best_conf:
                    best_conf = float(conf[i])
                    best_i = i
            if best_i < 0:
                self._pub_debug(debug_image, rgb_msg)
                continue

            mask = masks[best_i]
            if mask.shape[:2] != rgb_image.shape[:2]:
                mask = cv2.resize(mask, (rgb_image.shape[1], rgb_image.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            binary = (mask > 0.5).astype(np.uint8)
            mom = cv2.moments(binary)
            if mom["m00"] == 0:
                self._pub_debug(debug_image, rgb_msg)
                continue

            cu = int(mom["m10"] / mom["m00"])
            cv_ = int(mom["m01"] / mom["m00"])

            dm = depth_image.copy()
            try:
                dm[binary == 0] = 0
            except Exception:
                self._pub_debug(debug_image, rgb_msg)
                continue

            vd = dm[dm > 0]
            if vd is None or len(vd) < 10:
                self._pub_debug(debug_image, rgb_msg)
                continue

            depth = float(np.median(vd))
            if depth > 100.0:
                depth /= 1000.0
            if not (self.min_depth_m <= depth <= self.max_depth_m):
                self._pub_debug(debug_image, rgb_msg)
                continue

            x_cam = (cu - self.cx) * depth / self.fx
            y_cam = (cv_ - self.cy) * depth / self.fy

            ps = PoseStamped()
            ps.header = rgb_msg.header
            ps.header.frame_id = "camera_color_frame"
            ps.pose.position.x = float(x_cam)
            ps.pose.position.y = float(y_cam)
            ps.pose.position.z = float(depth)
            ps.pose.orientation.w = 1.0

            try:
                tf = self.tf_buffer.lookup_transform(
                    BASE_FRAME,
                    "camera_color_frame",
                    rclpy.time.Time(),
                    timeout=Duration(seconds=0.5),
                )
                pw = do_transform_pose_stamped(ps, tf)
            except Exception:
                self._pub_debug(debug_image, rgb_msg)
                continue

            wx = float(pw.pose.position.x)
            wy = float(pw.pose.position.y)
            wz = float(pw.pose.position.z)
            if not self._in_ws(wx, wy, wz):
                self._log_rejected_pose(wx, wy, wz, "outside workspace bounds")
                self._pub_debug(debug_image, rgb_msg)
                continue

            with self.detection_lock:
                if self.tracked_pose is None:
                    self.tracked_pose = (wx, wy, wz)
                    self.get_logger().info(
                        f"First valid detection: ({wx:.3f}, {wy:.3f}, {wz:.3f})"
                    )
                else:
                    a = self.smoothing_alpha
                    self.tracked_pose = (
                        a * wx + (1 - a) * self.tracked_pose[0],
                        a * wy + (1 - a) * self.tracked_pose[1],
                        a * wz + (1 - a) * self.tracked_pose[2],
                    )
                self.pose_timestamp = time.time()

            ov = np.zeros_like(debug_image)
            ov[binary > 0] = (0, 255, 0)
            debug_image = cv2.addWeighted(debug_image, 1.0, ov, 0.4, 0)
            cv2.circle(debug_image, (cu, cv_), 7, (0, 0, 255), -1)
            x1, y1, x2, y2 = map(int, xyxy[best_i])
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            with self.detection_lock:
                tp = self.tracked_pose
            if tp is not None:
                cv2.putText(debug_image, f"({tp[0]:.3f},{tp[1]:.3f},{tp[2]:.3f})",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 255), 1)
            self._pub_debug(debug_image, rgb_msg)

    def _get_pose_if_fresh(self, max_age_s: float) -> Optional[Tuple[float, float, float]]:
        with self.detection_lock:
            if self.tracked_pose is None or self.pose_timestamp is None:
                return None
            if (time.time() - self.pose_timestamp) > max_age_s:
                return None
            return tuple(self.tracked_pose)

    def _clear_tracking(self) -> None:
        with self.detection_lock:
            self.tracked_pose = None
            self.pose_timestamp = None

    def _frame_position(self, frame_name: str, timeout_s: float = 0.2) -> Optional[Tuple[float, float, float]]:
        try:
            tf_frame = self.tf_buffer.lookup_transform(
                BASE_FRAME,
                frame_name,
                rclpy.time.Time(),
                timeout=Duration(seconds=timeout_s),
            )
        except Exception:
            return None
        return (
            float(tf_frame.transform.translation.x),
            float(tf_frame.transform.translation.y),
            float(tf_frame.transform.translation.z),
        )

    def _gripper_midpoint_to_active_bottle_distance(self) -> Optional[float]:
        bottle_pose = self._active_bottle_pose
        if bottle_pose is None:
            return None
        left_tip = self._frame_position(LEFT_FINGER_TIP_FRAME)
        right_tip = self._frame_position(RIGHT_FINGER_TIP_FRAME)
        if left_tip is None or right_tip is None:
            return None
        midpoint = (
            0.5 * (left_tip[0] + right_tip[0]),
            0.5 * (left_tip[1] + right_tip[1]),
            0.5 * (left_tip[2] + right_tip[2]),
        )
        return math.sqrt(sum((midpoint[i] - bottle_pose[i]) ** 2 for i in range(3)))

    def _should_ignore_obstacle_for_bottle(self) -> bool:
        if self.obstacle_ignore_bottle_within_m <= 0.0:
            return False
        dist = self._gripper_midpoint_to_active_bottle_distance()
        if dist is None or dist > self.obstacle_ignore_bottle_within_m:
            return False
        now = time.monotonic()
        if (now - self._obstacle_ignore_last_log_mono) >= 0.5:
            self._obstacle_ignore_last_log_mono = now
            self.get_logger().info(
                f"[OBS] ignoring obstacle gate: bottle is {dist:.3f}m from gripper midpoint "
                f"(threshold {self.obstacle_ignore_bottle_within_m:.3f}m)"
            )
        return True

    # ─────────────────────────────────────────────────────
    # Motion helpers (joint-space + cartesian LIN)
    # ─────────────────────────────────────────────────────

    def _arm_joint_target(self, joints_raw: List[float], label: str) -> List[float]:
        if len(joints_raw) < len(ARM_JOINTS):
            raise ValueError(
                f"{label} has {len(joints_raw)} values, expected at least {len(ARM_JOINTS)}")
        if len(joints_raw) > len(ARM_JOINTS):
            self.get_logger().info(
                f"{label}: using first {len(ARM_JOINTS)} values for manipulator")
        return [float(v) for v in joints_raw[:len(ARM_JOINTS)]]

    def _log_motion_diag(
        self,
        label: str,
        target_pos: Optional[List[float]] = None,
        target_quat: Optional[List[float]] = None,
    ) -> None:
        try:
            st = self.moveit.query_state().name
        except Exception as e:
            st = f"<query_failed:{e}>"
        msg = [f"[DIAG] {label}: moveit_state={st}"]
        if target_pos is not None:
            msg.append(
                f"target_pos=({target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f})"
            )
        if target_quat is not None:
            msg.append(
                f"target_quat=({target_quat[0]:+.3f},{target_quat[1]:+.3f},"
                f"{target_quat[2]:+.3f},{target_quat[3]:+.3f})"
            )
        try:
            tf_tip = self.tf_buffer.lookup_transform(
                BASE_FRAME,
                self._tip_link,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.2),
            )
            cx = float(tf_tip.transform.translation.x)
            cy = float(tf_tip.transform.translation.y)
            cz = float(tf_tip.transform.translation.z)
            msg.append(f"ee_pos=({cx:.3f},{cy:.3f},{cz:.3f})")
            if target_pos is not None:
                msg.append(
                    f"delta=({target_pos[0]-cx:+.3f},{target_pos[1]-cy:+.3f},{target_pos[2]-cz:+.3f})"
                )
        except Exception as e:
            msg.append(f"ee_lookup_failed={e}")
        self.get_logger().warn(" ".join(msg))

    def _wait_for_motion_result(self, label: str, timeout_s: float = 60.0,
                                requesting_timeout_s: float = 10.0) -> bool:
        t0 = time.time()
        seen_active = False
        last_state = None
        last_heartbeat_t = 0.0
        requesting_t0 = None
        self.get_logger().info(f"[MOVE] waiting for motion result: {label}")
        while rclpy.ok():
            if self._obstacle_gate_active and self._obstacle_is_blocked():
                closest, _, min_seen, age = self._obstacle_snapshot()
                self.get_logger().error(
                    f"[OBS] blocked during {label}: "
                    f"closest={self._fmt_depth_m(closest)} "
                    f"min_seen={self._fmt_depth_m(min_seen)} age={age:.2f}s")
                self._last_motion_abort_reason = "obstacle_blocked"
                self._abort_motion_request(label)
                self._log_motion_diag(label)
                return False

            state = self.moveit.query_state()
            if state != last_state:
                self.get_logger().info(f"[MOVE] {label} state={state.name}")
                last_state = state
            if state != MoveIt2State.IDLE:
                seen_active = True
                # Track how long we've been stuck in REQUESTING
                if state == MoveIt2State.REQUESTING:
                    if requesting_t0 is None:
                        requesting_t0 = time.time()
                    elif (time.time() - requesting_t0) > requesting_timeout_s:
                        self.get_logger().error(
                            f"[MOVE] {label} stuck in REQUESTING for "
                            f"{requesting_timeout_s:.0f}s, cancelling goal")
                        self._log_motion_diag(label)
                        self._last_motion_abort_reason = "requesting_timeout"
                        self._abort_motion_request(label)
                        time.sleep(0.5)
                        return False
                else:
                    requesting_t0 = None
            elif seen_active:
                ok = bool(self.moveit.motion_suceeded)
                self.get_logger().info(f"[MOVE] {label} completed success={int(ok)}")
                if not ok:
                    self._log_motion_diag(label)
                return ok
            elif (time.time() - t0) > 0.25:
                self.get_logger().warn(
                    f"[MOVE] {label} never became active; treating as failed")
                self._last_motion_abort_reason = "never_active"
                self._log_motion_diag(label)
                return False

            now = time.time()
            if seen_active and (now - last_heartbeat_t) > 1.0:
                if self._obstacle_gate_active:
                    closest, blocked, min_seen, age = self._obstacle_snapshot()
                    self.get_logger().info(
                        f"[MOVE] {label} still waiting t={now - t0:.1f}s state={state.name} "
                        f"shortest_depth={self._fmt_depth_m(closest)} "
                        f"blocked={int(blocked)} min_seen={self._fmt_depth_m(min_seen)} "
                        f"depth_age={age:.2f}s")
                else:
                    self.get_logger().info(
                        f"[MOVE] {label} still waiting t={now - t0:.1f}s state={state.name}")
                last_heartbeat_t = now

            if (time.time() - t0) > timeout_s:
                self.get_logger().error(
                    f"[MOVE] {label} timeout after {timeout_s:.1f}s state={state.name}")
                self._last_motion_abort_reason = "motion_timeout"
                self._abort_motion_request(label)
                self._log_motion_diag(label)
                return False
            time.sleep(0.02)

    def _abort_motion_request(self, label: str) -> None:
        """Best-effort stop/reset of MoveIt2 local execution state."""
        try:
            st = self.moveit.query_state()
        except Exception:
            st = None
        if st == MoveIt2State.EXECUTING:
            try:
                self.moveit.cancel_execution()
            except Exception:
                pass
        try:
            self.moveit.force_reset_executing_state()
        except Exception:
            pass
        self.get_logger().warn(f"[MOVE] {label} motion request reset")

    def _move_joints(self, joints: List[float], label: str) -> bool:
        try:
            self._last_motion_abort_reason = ""
            self.get_logger().info(f"[MOVE] joints {label}: {np.round(joints, 3).tolist()}")
            self.moveit.move_to_configuration(joints)
            ok = self._wait_for_motion_result(label)
            if not ok:
                self.get_logger().error(f"[MOVE] joints {label} failed (plan/execute)")
            return ok
        except Exception as e:
            self.get_logger().error(f"[MOVE] joints {label} failed: {e}")
            return False

    def _move_pose(
        self,
        pos: List[float],
        quat: List[float],
        label: str,
        cartesian: bool = False,
    ) -> bool:
        """Move to a pose in base_link."""
        try:
            self._last_motion_abort_reason = ""
            self.get_logger().info(
                f"[MOVE] {label}: pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) "
                f"quat=({quat[0]:+.3f},{quat[1]:+.3f},{quat[2]:+.3f},{quat[3]:+.3f}) "
                f"cartesian={int(cartesian)}")
            self.moveit.move_to_pose(position=pos, quat_xyzw=quat, cartesian=cartesian)
            ok = self._wait_for_motion_result(label)
            if not ok:
                self._log_motion_diag(label, pos, quat)
                self.get_logger().error(f"[MOVE] {label} failed (plan/execute)")
            return ok
        except Exception as e:
            self._log_motion_diag(label, pos, quat)
            self.get_logger().error(f"[MOVE] {label} failed: {e}")
            return False

    @staticmethod
    def _approach_vector(quat_xyzw: List[float]) -> List[float]:
        """Return the EE approach axis (local +Z) expressed in base frame."""
        x, y, z, w = quat_xyzw
        # Third column of the rotation matrix from quaternion
        ax = 2.0 * (x * z + w * y)
        ay = 2.0 * (y * z - w * x)
        az = 1.0 - 2.0 * (x * x + y * y)
        return [ax, ay, az]

    # ─────────────────────────────────────────────────────
    # Obstacle-aware motion helper for grasp approach
    # ─────────────────────────────────────────────────────

    def _obstacle_aware_move(
        self,
        pos: List[float],
        quat: List[float],
        label: str,
        cartesian: bool = False,
    ) -> bool:
        """
        Attempt a move with variant-specific obstacle handling.
        Returns True if the move succeeded, False if it permanently failed.
        Raises _ObstacleAbort if the variant dictates a full task abort.
        """
        while rclpy.ok():
            if self._move_pose(pos, quat, label, cartesian=cartesian):
                return True

            if self._last_motion_abort_reason != "obstacle_blocked":
                return False

            # Variant-specific obstacle behavior
            if self.variant == "feas_only":
                self._task_result = (
                    False, "feas_only abort: obstacle detected during grasp approach")
                raise _ObstacleAbort()

            if self.variant == "hold_commit":
                self.get_logger().warn(
                    f"[OBS] hold_commit: obstacle during {label}; "
                    f"pausing {self.hold_commit_obstacle_pause_sec:.1f}s then retrying")
                time.sleep(max(0.0, self.hold_commit_obstacle_pause_sec))
                continue

            if self.variant == "hac":
                self.get_logger().warn(
                    f"[OBS] hac: obstacle during {label}; "
                    "waiting for clear before retry")
                while rclpy.ok() and self._obstacle_is_blocked():
                    time.sleep(0.10)
                if not rclpy.ok():
                    self._task_result = (
                        False, "Interrupted while waiting for obstacle to clear")
                    raise _ObstacleAbort()
                self.get_logger().info(f"[OBS] hac: clear, retrying {label}")
                continue

            return False

    # ─────────────────────────────────────────────────────
    # Task execution
    # ─────────────────────────────────────────────────────

    def _run_task(self):
        try:
            self.get_logger().info("[TASK] starting bottle pick task")
            self._set_state("WAIT_CAMERA")
            self._mark_started()

            # 1. Wait for camera intrinsics
            t0 = time.time()
            while rclpy.ok() and self.camera_info is None:
                if (time.time() - t0) > 20.0:
                    self._task_result = (False, "Camera intrinsics timeout")
                    return
                time.sleep(0.05)

            # 2. Open gripper
            self._set_state("OPEN_GRIPPER")
            if not self.gripper.command(self.gripper_open):
                self._task_result = (False, "Gripper open failed")
                return
            time.sleep(0.2)

            # 3. Cycle through scan points looking for the bottle
            self._set_state("SCAN")
            bottle_pose = None

            for cycle in range(self.max_scan_cycles):
                if bottle_pose is not None:
                    break
                for scan_idx, scan_joints in enumerate(SCAN_POSES):
                    label = f"scan_c{cycle}_p{scan_idx}"
                    self.get_logger().info(
                        f"[SCAN] cycle {cycle+1}/{self.max_scan_cycles}, "
                        f"point {scan_idx+1}/{len(SCAN_POSES)}")
                    self._clear_tracking()

                    if not self._move_joints(scan_joints, label):
                        self.get_logger().warn(f"[SCAN] {label} move failed, skipping")
                        continue

                    # Dwell at this scan point and check for detections
                    self.get_logger().info(
                        f"[SCAN] dwelling {self.scan_dwell_s:.1f}s at point {scan_idx+1}")
                    dwell_t0 = time.time()
                    while rclpy.ok() and (time.time() - dwell_t0) < self.scan_dwell_s:
                        bottle_pose = self._get_pose_if_fresh(self.pose_timeout_s)
                        if bottle_pose is not None:
                            self.get_logger().info(
                                f"[SCAN] bottle detected at "
                                f"({bottle_pose[0]:.3f},{bottle_pose[1]:.3f},{bottle_pose[2]:.3f}) "
                                f"during scan point {scan_idx+1}")
                            break
                        time.sleep(0.05)

                    if bottle_pose is not None:
                        break

            if bottle_pose is None:
                self._task_result = (False, "Bottle not detected after all scan cycles")
                return

            self.t_detect_s = self._trial_t()
            self.get_logger().info(
                f"[DETECT] bottle_filtered="
                f"({bottle_pose[0]:.3f},{bottle_pose[1]:.3f},{bottle_pose[2]:.3f})")
            if bottle_pose[0] < 0.0:
                self.get_logger().warn(
                    f"[DIAG] detected bottle has x<0 ({bottle_pose[0]:.3f}); "
                    "this is behind base frame and often causes self-collision/red state")
            self._publish_bottle_marker(bottle_pose)

            # 4. Compute grasp target (apply offsets)
            grasp_target = [
                float(bottle_pose[0]) + self.grasp_offset_x_m,
                float(bottle_pose[1]) + self.grasp_offset_y_m,
                float(bottle_pose[2]) + self.grasp_offset_z_m,
            ]
            # Enforce minimum Z
            if grasp_target[2] < self.min_grasp_z_m:
                grasp_target[2] = self.min_grasp_z_m

            # 5. Move to grasp: optional standoff, then a standard MoveIt pose move.
            #    No cartesian LIN segment is used here.
            self._set_state("MOVE_GRASP")
            self._active_bottle_pose = tuple(float(v) for v in bottle_pose)
            standoff_dist = self.grasp_lin_approach_m

            # Get current EE pose (orientation from scan position)
            try:
                tf_ee = self.tf_buffer.lookup_transform(
                    BASE_FRAME,
                    self._tip_link,
                    rclpy.time.Time(),
                    timeout=Duration(seconds=1.0),
                )
                ee_quat = [
                    float(tf_ee.transform.rotation.x),
                    float(tf_ee.transform.rotation.y),
                    float(tf_ee.transform.rotation.z),
                    float(tf_ee.transform.rotation.w),
                ]
            except Exception as e:
                self._task_result = (False, f"EE TF lookup failed: {e}")
                return

            # Approach vector = EE local +Z in base frame
            approach = self._approach_vector(ee_quat)

            # Standoff = grasp target pulled back along -approach
            standoff_target = [
                grasp_target[0] - standoff_dist * approach[0],
                grasp_target[1] - standoff_dist * approach[1],
                grasp_target[2] - standoff_dist * approach[2],
            ]

            self.get_logger().info(
                f"[MOVE] grasp_target="
                f"({grasp_target[0]:.3f},{grasp_target[1]:.3f},{grasp_target[2]:.3f}) "
                f"ee_quat=({ee_quat[0]:+.3f},{ee_quat[1]:+.3f},"
                f"{ee_quat[2]:+.3f},{ee_quat[3]:+.3f}) "
                f"approach=({approach[0]:+.3f},{approach[1]:+.3f},{approach[2]:+.3f}) "
                f"standoff=({standoff_target[0]:.3f},{standoff_target[1]:.3f},"
                f"{standoff_target[2]:.3f}) standoff_dist={standoff_dist:.3f}m")

            grasp_reached = False
            if self._obstacle_gate_enabled:
                self._set_obstacle_gate(True, "move_grasp_start")
            try:
                if standoff_dist > 1e-4:
                    # Phase 1: Move to a pulled-back standoff pose.
                    self.get_logger().info("[MOVE] Phase 1: MoveIt move to standoff")
                    try:
                        standoff_ok = self._obstacle_aware_move(
                            standoff_target, ee_quat, "standoff", cartesian=False)
                    except _ObstacleAbort:
                        return
                    if not standoff_ok:
                        self._task_result = (False, "Could not reach standoff pose")
                        return

                self.last_grasp_quat = list(ee_quat)

                # Phase 2: MoveIt pose moves in small increments to the grasp target.
                current_ee_pos = self._frame_position(self._tip_link, timeout_s=0.5)
                if current_ee_pos is None:
                    self._task_result = (False, "Could not read EE pose before grasp steps")
                    return

                dist_to_target = math.sqrt(sum(
                    (grasp_target[i] - current_ee_pos[i]) ** 2 for i in range(3)
                ))
                step_size = max(1e-4, self.grasp_step_m)
                num_steps = max(1, int(math.ceil(dist_to_target / step_size)))
                self.get_logger().info(
                    f"[MOVE] Phase 2: MoveIt grasp steps to target "
                    f"(distance={dist_to_target:.3f}m, step={step_size:.3f}m, count={num_steps})")

                grasp_abort_reason = ""
                for step_idx in range(1, num_steps + 1):
                    frac = float(step_idx) / float(num_steps)
                    step_target = [
                        current_ee_pos[0] + frac * (grasp_target[0] - current_ee_pos[0]),
                        current_ee_pos[1] + frac * (grasp_target[1] - current_ee_pos[1]),
                        current_ee_pos[2] + frac * (grasp_target[2] - current_ee_pos[2]),
                    ]
                    self.get_logger().info(
                        f"[MOVE] grasp step {step_idx}/{num_steps}: "
                        f"({step_target[0]:.3f},{step_target[1]:.3f},{step_target[2]:.3f})")
                    try:
                        step_ok = self._obstacle_aware_move(
                            step_target, ee_quat, f"grasp_step_{step_idx}", cartesian=False)
                    except _ObstacleAbort:
                        return
                    grasp_abort_reason = str(self._last_motion_abort_reason)
                    if not step_ok:
                        break
                else:
                    grasp_reached = True
                    self.get_logger().info("[MOVE] grasp pose reached")

                if not grasp_reached:
                    disallowed_abort_reasons = {"requesting_timeout", "never_active"}
                    if grasp_abort_reason in disallowed_abort_reasons:
                        self.get_logger().error(
                            "[MOVE] grasp move never became executable; "
                            "refusing close-enough fallback")
                    else:
                        # Grasp move failed — check if EE is already close
                        # enough to the grasp target to proceed anyway.
                        try:
                            tf_now = self.tf_buffer.lookup_transform(
                                BASE_FRAME, self._tip_link,
                                rclpy.time.Time(),
                                timeout=Duration(seconds=0.5),
                            )
                            ee_now = [
                                float(tf_now.transform.translation.x),
                                float(tf_now.transform.translation.y),
                                float(tf_now.transform.translation.z),
                            ]
                            dist = math.sqrt(sum(
                                (ee_now[i] - grasp_target[i]) ** 2
                                for i in range(3)
                            ))
                            close_enough_m = max(0.0, self.grasp_close_enough_m)
                            self.get_logger().info(
                                f"[MOVE] EE dist to grasp target: {dist:.4f}m "
                                f"(threshold: {close_enough_m:.4f}m)")
                            if close_enough_m > 0.0 and dist <= close_enough_m:
                                grasp_reached = True
                                self.get_logger().warn(
                                    f"[MOVE] grasp move failed but EE is "
                                    f"{dist:.4f}m from target — close enough, "
                                    f"proceeding to grasp")
                        except Exception as e:
                            self.get_logger().warn(
                                f"[MOVE] close-enough check failed: {e}")

            finally:
                self._active_bottle_pose = None
                if self._obstacle_gate_active:
                    self._set_obstacle_gate(False, "move_grasp_end")

            if not grasp_reached:
                self._task_result = (False, "Could not reach grasp pose")
                return

            # 6. Close gripper 
            self._set_state("GRASP")
            if not self.gripper.command(self.gripper_closed):
                self._task_result = (False, "Gripper close failed")
                return
            self.t_grasp_s = self._trial_t()
            time.sleep(0.3)

            # 7. Lift bottle ~2 inches straight up from grasp pose.
            LIFT_M = 2.0 * INCH_TO_M  # 0.0508 m
            lift_target = [
                grasp_target[0],
                grasp_target[1],
                grasp_target[2] + LIFT_M,
            ]
            self._set_state("LIFT")
            self.get_logger().info(
                f"[MOVE] lifting bottle to z={lift_target[2]:.3f} "
                f"(+{LIFT_M:.3f}m from grasp)")
            if not self._move_pose(lift_target, self.last_grasp_quat, "lift_bottle"):
                self.get_logger().warn("[MOVE] lift failed, continuing to retreat anyway")

            # 8. Retreat to a safe intermediate scan pose.
            self._set_state("RETREAT")
            retreat_ok = False
            for ri, retreat_joints in enumerate(SCAN_POSES):
                label = f"retreat_{ri}"
                self.get_logger().info(
                    f"[MOVE] trying retreat waypoint {ri+1}/{len(SCAN_POSES)}")
                if self._move_joints(retreat_joints, label):
                    retreat_ok = True
                    break
            if not retreat_ok:
                self._task_result = (False, "Could not retreat to any safe pose after grasp")
                return

            # 9. Move to place joints (any path — this aligns the arm at the
            #    target config, OMPL can take whatever route it finds).
            self._set_state("MOVE_PLACE")
            place_ok = False
            for attempt, pt in enumerate([
                self.planning_time_s,
                self.planning_time_s * 2,
                self.planning_time_s * 4,
            ]):
                self.get_logger().info(
                    f"[MOVE] place_joints attempt {attempt+1}/3 "
                    f"planning_time={pt:.1f}s")
                prev_time = self.moveit.allowed_planning_time
                prev_attempts = self.moveit.num_planning_attempts
                self.moveit.allowed_planning_time = pt
                self.moveit.num_planning_attempts = max(
                    self.planning_attempts, 20 * (attempt + 1))
                try:
                    if self._move_joints(self.place_joints, f"place_joints_a{attempt}"):
                        place_ok = True
                        break
                finally:
                    self.moveit.allowed_planning_time = prev_time
                    self.moveit.num_planning_attempts = prev_attempts
            if not place_ok:
                self._task_result = (False, "Move to place joints failed (all attempts)")
                return

            # 10. After reaching place_joints, move in base +Y by
            #     place_forward_distance_m.
            self._set_state("PLACE_FORWARD")
            self._clear_bottle_marker()
            self._set_state("DONE")
            self._task_result = (True, "Reached place pose before forward move")
            return
        except _ObstacleAbort:
            pass  # _task_result already set by _obstacle_aware_move
        except Exception as e:
            self._task_result = (False, f"Exception: {e}")


class _ObstacleAbort(Exception):
    """Sentinel exception for feas_only abort during obstacle-aware moves."""
    pass


# ─────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = BottlePickSupervisorMetrics()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        try:
            node._finish("interrupted", "keyboard_interrupt")
        except Exception:
            pass
    finally:
        try:
            node._stop_event.set()
            node._new_pair_event.set()
            if hasattr(node, "_worker_thread") and node._worker_thread.is_alive():
                node._worker_thread.join(timeout=2.0)
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
