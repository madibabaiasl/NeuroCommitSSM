#!/usr/bin/env python3
"""
hold_to_rest_supervisor.py (minimal)

Behavior:
- Read a CSV with columns: center_time_s, commit_state
- Follow the CSV timeline from node start
- Whenever state transitions into HOLD, move robot arm to rest joints

No obstacle logic. No perception.
"""

import csv
import math
import threading
import time
from bisect import bisect_right
from copy import deepcopy
from typing import List, Tuple

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy, PlanRequestParameters
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_msgs.msg import MoveItErrorCodes, CollisionObject as CollisionObjectMsg
from shape_msgs.msg import SolidPrimitive


DEFAULT_REST_JOINTS = [
    1.8688011232975643,
    -0.35301318776658164,
    2.5798970922818016,
    0.23751190793460158,
    -1.4676909799280429,
    -1.6853127863005222,
    0.007074257344007493,
]


def _safe_float(v, default=float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _read_timeline_csv(
    pred_csv: str,
    time_col: str = "center_time_s",
    state_col: str = "commit_state",
) -> Tuple[List[float], List[str], List[int]]:
    with open(pred_csv, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    entries = []
    for i, row in enumerate(rows):
        t = _safe_float(row.get(time_col))
        s = str(row.get(state_col, "")).strip().upper()
        if math.isfinite(t) and s:
            entries.append((float(t), s, i + 2))  # +2: include header row offset

    if not entries:
        raise ValueError(
            f"No valid rows with finite '{time_col}' and non-empty '{state_col}'"
        )

    entries.sort(key=lambda x: x[0])
    times = [e[0] for e in entries]
    states = [e[1] for e in entries]
    rows = [e[2] for e in entries]
    return times, states, rows


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


class HoldToRestSupervisor(Node):
    def __init__(self):
        super().__init__("hold_to_rest_supervisor")

        self.declare_parameter("pred_csv", "")
        self.declare_parameter("time_col", "center_time_s")
        self.declare_parameter("state_col", "commit_state")
        self.declare_parameter("hold_value", "HOLD")
        self.declare_parameter("poll_period_s", 0.05)

        self.declare_parameter("rest_joints", DEFAULT_REST_JOINTS)
        self.declare_parameter(
            "trajectory_action_name", "/joint_trajectory_controller/follow_joint_trajectory"
        )
        self.declare_parameter("controller_wait_timeout_s", 8.0)

        self.declare_parameter("robot_name", "kinova_gen3_6dof_robotiq_2f_85")
        self.declare_parameter(
            "moveit_config_pkg", "kinova_gen3_6dof_robotiq_2f_85_moveit_config"
        )
        self.declare_parameter("move_group_name", "manipulator")
        self.declare_parameter("max_velocity_scaling", 0.30)
        self.declare_parameter("max_acceleration_scaling", 0.30)
        self.declare_parameter("collision_enabled", True)
        self.declare_parameter("table_point", [-0.5156, 0.5, 0.16105])
        self.declare_parameter("table_yaw", math.pi / 2.0)
        self.declare_parameter("wheelchair_point", [0.0, -0.39, 0.12065])
        self.declare_parameter("wheelchair_yaw", 0.0)

        self.pred_csv = str(self.get_parameter("pred_csv").value)
        self.time_col = str(self.get_parameter("time_col").value)
        self.state_col = str(self.get_parameter("state_col").value)
        self.hold_value = str(self.get_parameter("hold_value").value).strip().upper()
        self.poll_period_s = max(0.01, float(self.get_parameter("poll_period_s").value))

        self.rest_joints = [float(x) for x in list(self.get_parameter("rest_joints").value)]
        self.trajectory_action_name = str(self.get_parameter("trajectory_action_name").value)
        self.controller_wait_timeout_s = max(
            0.0, float(self.get_parameter("controller_wait_timeout_s").value)
        )

        self.max_vel = float(self.get_parameter("max_velocity_scaling").value)
        self.max_acc = float(self.get_parameter("max_acceleration_scaling").value)
        self.group_name = str(self.get_parameter("move_group_name").value)

        if not self.pred_csv:
            raise ValueError("pred_csv is required")

        self.times, self.states, self.src_rows = _read_timeline_csv(
            self.pred_csv, self.time_col, self.state_col
        )
        hold_count = sum(1 for s in self.states if s == self.hold_value)
        self.get_logger().info(
            f"Timeline loaded: {len(self.times)} rows, {hold_count} rows with "
            f"state={self.hold_value}, time range=[{self.times[0]:.3f}, {self.times[-1]:.3f}]s"
        )

        rn = str(self.get_parameter("robot_name").value)
        pkg = str(self.get_parameter("moveit_config_pkg").value)
        cfg = MoveItConfigsBuilder(rn, package_name=pkg).to_moveit_configs().to_dict()
        cfg = _augment_moveit_cfg(cfg)

        self._moveit = MoveItPy(node_name="hold_rest_moveitpy", config_dict=cfg)
        self._pc = self._moveit.get_planning_component(self.group_name)
        self._default_pipeline = str(cfg.get("default_planning_pipeline", "ompl"))

        self._traj_action_cli = ActionClient(
            self, FollowJointTrajectory, self.trajectory_action_name
        )

        self._t0 = time.monotonic()
        self._last_idx = -1
        self._prev_state = ""
        self._move_in_progress = False
        self._move_lock = threading.Lock()
        self._scene_collision_lock = threading.Lock()

        self._collision_refresh_timer = None
        if bool(self.get_parameter("collision_enabled").value):
            self._setup_collision_objects(log=True, blocking=True)
            self._collision_refresh_timer = self.create_timer(
                0.2, self._refresh_collision_objects
            )

        self.create_timer(self.poll_period_s, self._tick)
        self.get_logger().info(
            f"HoldToRestSupervisor up: pred_csv={self.pred_csv} "
            f"controller_action={self.trajectory_action_name}"
        )

    def _trial_t(self) -> float:
        return time.monotonic() - self._t0

    @staticmethod
    def _make_box(size: List[float]) -> SolidPrimitive:
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = list(size)
        return box

    @staticmethod
    def _make_collision_pose(position: List[float], yaw: float = 0.0) -> Pose:
        p = Pose()
        p.position.x, p.position.y, p.position.z = position
        half = yaw / 2.0
        p.orientation.x = 0.0
        p.orientation.y = 0.0
        p.orientation.z = math.sin(half)
        p.orientation.w = math.cos(half)
        return p

    def _build_collision_objects(self) -> Tuple[CollisionObjectMsg, CollisionObjectMsg]:
        table_point = list(self.get_parameter("table_point").value)
        table_yaw = float(self.get_parameter("table_yaw").value)
        wheelchair_point = list(self.get_parameter("wheelchair_point").value)
        wheelchair_yaw = float(self.get_parameter("wheelchair_yaw").value)

        # Same coarse sizes used by plant supervisor.
        table_size = [0.762 + 0.0127, 1.8288, 0.2921 + 0.0127]
        wheel_size = [0.51, 0.5, 0.2413]

        co_table = CollisionObjectMsg()
        co_table.id = "table_shelf"
        co_table.header.frame_id = "base_link"
        co_table.operation = CollisionObjectMsg.ADD
        co_table.primitives.append(self._make_box(table_size))
        co_table.primitive_poses.append(
            self._make_collision_pose(table_point, table_yaw)
        )

        co_wheel = CollisionObjectMsg()
        co_wheel.id = "wheelchair"
        co_wheel.header.frame_id = "base_link"
        co_wheel.operation = CollisionObjectMsg.ADD
        co_wheel.primitives.append(self._make_box(wheel_size))
        co_wheel.primitive_poses.append(
            self._make_collision_pose(wheelchair_point, wheelchair_yaw)
        )
        return co_table, co_wheel

    def _setup_collision_objects(self, log: bool = True, blocking: bool = True) -> bool:
        if self._moveit is None or not bool(self.get_parameter("collision_enabled").value):
            return False

        acquired = self._scene_collision_lock.acquire(blocking=blocking)
        if not acquired:
            return False

        try:
            co_table, co_wheel = self._build_collision_objects()
            with self._moveit.get_planning_scene_monitor().read_write() as scene:
                scene.apply_collision_object(co_table)
                scene.apply_collision_object(co_wheel)
            if log:
                self.get_logger().info(
                    "Collision objects applied: "
                    f"'table_shelf' ({len(co_table.primitives)}), "
                    f"'wheelchair' ({len(co_wheel.primitives)})"
                )
            return True
        except Exception as e:
            self.get_logger().error(f"Failed to setup collision objects: {e}")
            return False
        finally:
            self._scene_collision_lock.release()

    def _refresh_collision_objects(self):
        self._setup_collision_objects(log=False, blocking=False)

    def _execute_trajectory(self, trajectory) -> bool:
        if not self._traj_action_cli.wait_for_server(timeout_sec=self.controller_wait_timeout_s):
            self.get_logger().error(
                f"Controller action server not ready: {self.trajectory_action_name}"
            )
            return False

        res = self._moveit.execute(trajectory, controllers=[])
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

        return False

    def _plan_exec_joints(self, joints: List[float]) -> bool:
        try:
            self._pc.set_start_state_to_current_state()
            gs = RobotState(self._moveit.get_robot_model())
            gs.set_joint_group_positions(self.group_name, joints)
            gs.update()
            self._pc.set_goal_state(robot_state=gs)

            params = PlanRequestParameters(self._moveit, "")
            params.planning_pipeline = self._default_pipeline
            params.planning_time = 2.8
            params.planning_attempts = 1
            params.max_velocity_scaling_factor = self.max_vel
            params.max_acceleration_scaling_factor = self.max_acc

            result = self._pc.plan(single_plan_parameters=params)
            if not result:
                return False
            return self._execute_trajectory(result.trajectory)
        except Exception:
            return False

    def _move_to_rest_worker(self, reason: str):
        ok = False
        try:
            self.get_logger().info(f"[REST] HOLD detected -> move to rest ({reason})")
            candidates = [list(self.rest_joints)]
            if len(self.rest_joints) == 7:
                candidates.append(list(self.rest_joints[:6]))

            for i, target in enumerate(candidates, start=1):
                self.get_logger().info(
                    f"[REST] Attempt {i}: {len(target)} joint values"
                )
                if self._plan_exec_joints(target):
                    ok = True
                    break

            if ok:
                self.get_logger().info("[REST] Reached rest pose")
            else:
                self.get_logger().error("[REST] Failed to reach rest pose")
        finally:
            with self._move_lock:
                self._move_in_progress = False

    def _start_move_to_rest(self, reason: str):
        with self._move_lock:
            if self._move_in_progress:
                return
            self._move_in_progress = True

        threading.Thread(
            target=self._move_to_rest_worker,
            args=(reason,),
            daemon=True,
        ).start()

    def _tick(self):
        t = self._trial_t()
        idx = bisect_right(self.times, t) - 1
        if idx < 0:
            return

        state = self.states[idx]
        if idx != self._last_idx:
            self.get_logger().info(
                f"[CSV] t={t:.3f}s row={self.src_rows[idx]} csv_t={self.times[idx]:.3f}s state={state}"
            )
            self._last_idx = idx

        entered_hold = state == self.hold_value and self._prev_state != self.hold_value
        if entered_hold:
            self._start_move_to_rest(
                reason=f"row={self.src_rows[idx]} csv_t={self.times[idx]:.3f}s"
            )

        self._prev_state = state


def main():
    rclpy.init()
    node = HoldToRestSupervisor()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
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
