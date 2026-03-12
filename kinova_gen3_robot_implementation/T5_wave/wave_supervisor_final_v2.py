#!/usr/bin/env python3
"""
wave_supervisor_final.py  (ROS 2 Jazzy)

Features:
- Start timing driven by model CSV (commit_event) for commit_only / hold_commit / hac
- feas_only ignores commit timing (starts on stable(f_cv))
- Per-trial metric computation + output CSVs
- Flaps counted from pred_csv commit_state transitions (HOLD <-> COMMIT)
- Global metrics summary CSV appended after each run (single file across runs)

Run example:
  ros2 run bottle_grasping wave_supervisor_final --ros-args \
    -p variant:=commit_only \
    -p trial_id:=13_T5_commit_only \
    -p pred_csv:=/home/pascal/Downloads/Task-3-test/Task_5/13_T5_pred_2.csv \
    -p log_dir:=/tmp/wave_logs/13_T5_commit_only \
    -p global_summary_csv:=/tmp/wave_metrics_summary.csv
"""

import os
import csv
import time
import math
import threading
from typing import List, Optional, Dict, Any

import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from action_msgs.msg import GoalStatus
from moveit_msgs.msg import MoveItErrorCodes
from moveit_configs_utils import MoveItConfigsBuilder
from moveit.planning import MoveItPy, PlanRequestParameters
from moveit.core.robot_state import RobotState


def _augment_cfg(cfg: dict, jst: str = "/joint_states") -> dict:
    cfg = dict(cfg)
    pips = cfg.get("planning_pipelines", [])
    if isinstance(pips, dict):
        return cfg
    if not pips:
        pips = ["ompl"]
    default_p = cfg.get("default_planning_pipeline", pips[0])
    cfg["planning_pipelines"] = {"pipeline_names": pips}
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
            "planning_pipeline": default_p,
            "max_velocity_scaling_factor": 0.20,
            "max_acceleration_scaling_factor": 0.20,
            "planning_time": 2.5,
        },
    )
    return cfg


def _deg_to_rad(v: float) -> float:
    return math.radians(float(v))


def _lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _read_model_csv(pred_csv: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(pred_csv):
        raise FileNotFoundError(f"pred_csv not found: {pred_csv}")

    with open(pred_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"pred_csv is empty: {pred_csv}")

    required = ["center_time_s", "commit_event", "commit_state", "gt_action"]
    for k in required:
        if k not in rows[0]:
            raise KeyError(f"Missing required column '{k}' in {pred_csv}")

    center_time_s = np.array([_safe_float(r["center_time_s"]) for r in rows], dtype=np.float64)
    commit_event = np.array([int(float(r["commit_event"])) for r in rows], dtype=np.int32)
    commit_state = np.array([str(r["commit_state"]) for r in rows], dtype=object)
    gt_action = np.array([int(float(r["gt_action"])) for r in rows], dtype=np.int32)

    return {
        "center_time_s": center_time_s,
        "commit_event": commit_event,
        "commit_state": commit_state,
        "gt_action": gt_action,
    }


def _first_commit_time(model: Dict[str, np.ndarray]) -> Optional[float]:
    idx = np.where(model["commit_event"] == 1)[0]
    if idx.size == 0:
        return None
    return float(model["center_time_s"][int(idx[0])])


def _nearest_index(times: np.ndarray, t: float) -> int:
    return int(np.argmin(np.abs(times - float(t))))


def _count_flaps(states: List[str]) -> int:
    if not states:
        return 0
    flaps = 0
    prev = states[0]
    for s in states[1:]:
        if s != prev:
            flaps += 1
            prev = s
    return flaps


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


class WaveHACSupervisor(Node):
    def __init__(self):
        super().__init__("wave_hac_supervisor")

        self.declare_parameter("variant", "hac")
        self.declare_parameter("trial_id", "t0001")
        self.declare_parameter("log_dir", "/tmp/wave_hac_logs")
        self.declare_parameter("pred_csv", "")
        self.declare_parameter("global_summary_csv", "/tmp/wave_metrics_summary.csv")

        self.declare_parameter("depth_topic", "/camera/depth_registered/image_rect")
        self.declare_parameter("roi_fraction", 0.30)
        self.declare_parameter("min_valid_px", 80)
        self.declare_parameter("gate_dist_m", 0.60)
        self.declare_parameter("clear_dist_m", 0.65)
        self.declare_parameter("dwell_s", 0.25)
        self.declare_parameter("state_hz", 20.0)

        self.declare_parameter("robot_name", "kinova_gen3_6dof_robotiq_2f_85")
        self.declare_parameter("moveit_config_pkg", "kinova_gen3_6dof_robotiq_2f_85_moveit_config")
        self.declare_parameter("move_group_name", "manipulator")
        self.declare_parameter("rest_joints_rad", [])

        self.declare_parameter(
            "start_joints_rad",
            [1.619472, -0.306040, 2.232634, 0.145396, -2.038570, -1.651546],
        )

        self.declare_parameter(
            "wave_ready_joints_rad",
            [1.804020661887768,
             0.2686897012637274,
             1.2056839987219217,
             -0.13713362137923202,
             -0.8663848921467574,
             -1.6516174069244034],
        )

        self.declare_parameter("approach_steps", 10)
        self.declare_parameter("wave_joint_index", 3)
        self.declare_parameter("wave_min_deg", -52.0)
        self.declare_parameter("wave_max_deg", 52.0)
        self.declare_parameter("wave_cycles", 2)
        self.declare_parameter("pause_sec", 0.08)

        self.declare_parameter("planning_time", 3.0)
        self.declare_parameter("planning_attempts", 2)
        self.declare_parameter("max_velocity_scaling", 0.28)
        self.declare_parameter("max_acceleration_scaling", 0.28)

        self.variant = str(self.get_parameter("variant").value)
        self.trial_id = str(self.get_parameter("trial_id").value)
        self.log_dir = str(self.get_parameter("log_dir").value)
        self.pred_csv = str(self.get_parameter("pred_csv").value)
        self.global_summary_csv = str(self.get_parameter("global_summary_csv").value)

        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.roi_fraction = float(self.get_parameter("roi_fraction").value)
        self.min_valid_px = int(self.get_parameter("min_valid_px").value)
        self.gate_dist_m = float(self.get_parameter("gate_dist_m").value)
        self.clear_dist_m = float(self.get_parameter("clear_dist_m").value)
        self.dwell_s = float(self.get_parameter("dwell_s").value)

        self.group_name = str(self.get_parameter("move_group_name").value)
        self.start_joints_rad = [float(x) for x in self.get_parameter("start_joints_rad").value]
        self.wave_ready_joints_rad = [float(x) for x in self.get_parameter("wave_ready_joints_rad").value]
        self.approach_steps = int(self.get_parameter("approach_steps").value)

        rest = list(self.get_parameter("rest_joints_rad").value)
        self.rest_joints_rad = [float(x) for x in rest] if len(rest) == 6 else []

        self.wave_joint_index = int(self.get_parameter("wave_joint_index").value)
        self.wave_min_rad = _deg_to_rad(float(self.get_parameter("wave_min_deg").value))
        self.wave_max_rad = _deg_to_rad(float(self.get_parameter("wave_max_deg").value))
        self.wave_cycles = int(self.get_parameter("wave_cycles").value)
        self.pause_sec = float(self.get_parameter("pause_sec").value)

        self.planning_time = float(self.get_parameter("planning_time").value)
        self.planning_attempts = int(self.get_parameter("planning_attempts").value)
        self.max_vel = float(self.get_parameter("max_velocity_scaling").value)
        self.max_acc = float(self.get_parameter("max_acceleration_scaling").value)

        if len(self.start_joints_rad) != 6 or len(self.wave_ready_joints_rad) != 6:
            raise ValueError("start_joints_rad and wave_ready_joints_rad must have 6 joint values.")
        if self.approach_steps < 2:
            raise ValueError("approach_steps must be >= 2.")
        if self.wave_joint_index < 0 or self.wave_joint_index > 5:
            raise ValueError("wave_joint_index must be in [0..5].")

        self.model: Optional[Dict[str, np.ndarray]] = None
        self.t_commit_s: Optional[float] = None

        if self.pred_csv.strip():
            self.model = _read_model_csv(self.pred_csv)
            self.t_commit_s = _first_commit_time(self.model)
            if self.t_commit_s is None:
                self.get_logger().warn(f"No commit_event==1 found in pred_csv={self.pred_csv}.")
            else:
                self.get_logger().info(f"[MODEL] commit_event first at t_commit={self.t_commit_s:.3f}s (from {self.pred_csv})")
        else:
            self.get_logger().warn("pred_csv is empty.")

        self.bridge = CvBridge()
        self.t0_wall = time.time()

        self.closest_m = math.inf
        self.f_cv = 1
        self._feas_since: Optional[float] = None

        self.sup_state = "HOLD"
        self.started_flag = 0
        self.t_start_s = float("nan")
        self.t_success_s = float("nan")
        self.outcome = "timeout"

        self._phase = "IDLE"

        self._abort_requested = False
        self._pause_requested = False
        self._resume_event = threading.Event()
        self._resume_event.set()

        self._hold_commit_tripped = False
        self._hold_commit_ignore_cv = False

        os.makedirs(self.log_dir, exist_ok=True)
        self.attempts_path = os.path.join(self.log_dir, "attempts.csv")
        self.state_path = os.path.join(self.log_dir, "state_stream.csv")
        self.cv_path = os.path.join(self.log_dir, "cv_stream.csv")
        self.metrics_trial_path = os.path.join(self.log_dir, "metrics_trial.csv")

        self._init_csv(self.attempts_path, ["trial_id", "variant", "started_flag", "t_start_s", "t_success_s", "outcome"])
        self._init_csv(self.state_path, ["t_s", "trial_id", "variant", "sup_state"])
        self._init_csv(self.cv_path, ["t_s", "trial_id", "variant", "closest_m", "f_cv"])
        self._init_csv(
            self.metrics_trial_path,
            [
                "trial_id", "variant", "started_flag", "outcome",
                "t_start_s", "t_success_s", "time_to_success_s",
                "false_start", "gt_action_at_start",
                "cv_infeas_start", "f_cv_at_start",
                "flaps"
            ],
        )

        self.global_summary_header = [
            "trial_id",
            "variant",
            "log_dir",
            "pred_csv",
            "started_flag",
            "outcome",
            "t_start_s",
            "t_success_s",
            "false_start",
            "cv_infeas_start",
            "flaps",
            "time_to_success_s",
        ]
        self._init_global_summary(self.global_summary_csv, self.global_summary_header)

        self.create_subscription(Image, self.depth_topic, self._on_depth, 10)
        state_hz = float(self.get_parameter("state_hz").value)
        self.create_timer(1.0 / max(1.0, state_hz), self._log_state_tick)

        rn = str(self.get_parameter("robot_name").value)
        pkg = str(self.get_parameter("moveit_config_pkg").value)
        self.get_logger().info("Initializing MoveItPy ...")
        mcfg = MoveItConfigsBuilder(rn, package_name=pkg).to_moveit_configs()
        cfg = _augment_cfg(mcfg.to_dict(), jst="/joint_states")
        self._default_pipeline = str(cfg.get("default_planning_pipeline", "ompl"))
        self._moveit = MoveItPy(node_name=f"{self.get_name()}_moveitpy", config_dict=cfg)
        self._pc = self._moveit.get_planning_component(self.group_name)
        self.get_logger().info(f"MoveItPy ready. pipeline={self._default_pipeline} group={self.group_name}")

        self.get_logger().info(
            f"Variant={self.variant} trial_id={self.trial_id} "
            f"gate={self.gate_dist_m:.2f}m clear={self.clear_dist_m:.2f}m dwell={self.dwell_s:.2f}s "
            f"approach_steps={self.approach_steps} global_summary_csv={self.global_summary_csv}"
        )

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _t_s(self) -> float:
        return time.time() - self.t0_wall

    def _init_csv(self, path: str, header: List[str]) -> None:
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(header)

    def _init_global_summary(self, path: str, header: List[str]) -> None:
        _ensure_parent_dir(path)
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(header)

    def _append_csv(self, path: str, row: List) -> None:
        with open(path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def _log_state_tick(self):
        self._append_csv(self.state_path, [self._t_s(), self.trial_id, self.variant, self.sup_state])

    def _on_depth(self, msg: Image):
        try:
            depth_raw = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        except Exception as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return

        d = depth_raw.astype(np.float32)
        enc = (msg.encoding or "").lower()
        if enc in ("16uc1", "mono16"):
            d /= 1000.0

        if d.ndim < 2:
            return

        h, w = d.shape[:2]
        rh = int(h * self.roi_fraction / 2.0)
        rw = int(w * self.roi_fraction / 2.0)
        cy, cx = h // 2, w // 2
        roi = d[max(0, cy - rh):min(h, cy + rh), max(0, cx - rw):min(w, cx + rw)]

        valid = roi[np.isfinite(roi) & (roi > 0.01) & (roi < 5.0)]
        if valid.size < self.min_valid_px:
            closest = math.inf
        else:
            closest = float(np.percentile(valid, 5.0))

        self.closest_m = closest

        if self.f_cv == 1:
            if closest < self.gate_dist_m:
                self.f_cv = 0
                self._feas_since = None
        else:
            if closest > self.clear_dist_m:
                self.f_cv = 1
                self._feas_since = self._t_s()

        if self.f_cv == 1 and self._feas_since is None:
            self._feas_since = self._t_s()

        self._append_csv(self.cv_path, [self._t_s(), self.trial_id, self.variant, self.closest_m, self.f_cv])

        if self.started_flag == 1 and self._phase == "APPROACH" and self.variant == "hac":
            if self.f_cv == 0:
                self._request_pause()
            elif self._stable_feasible():
                self._request_resume()

        if self.started_flag == 1 and self._phase == "APPROACH" and self.variant == "feas_only":
            if self.f_cv == 0:
                self._request_abort("CV breach during APPROACH (feas_only)")

    def _stable_feasible(self) -> bool:
        if self.f_cv != 1 or self._feas_since is None:
            return False
        return (self._t_s() - self._feas_since) >= self.dwell_s

    def _request_abort(self, reason: str):
        if not self._abort_requested:
            self.get_logger().warn(f"[ABORT REQUEST] {reason}")
        self._abort_requested = True
        self._resume_event.set()

    def _request_pause(self):
        if not self._pause_requested:
            self.get_logger().info("[HAC] PAUSE requested (CV breach during APPROACH)")
        self._pause_requested = True
        self.sup_state = "PAUSE"
        self._resume_event.clear()

    def _request_resume(self):
        if self._pause_requested:
            self.get_logger().info("[HAC] RESUME (stable feasible)")
        self._pause_requested = False
        if self.started_flag == 1:
            self.sup_state = "EXECUTE"
        self._resume_event.set()

    def _wait_if_paused(self) -> bool:
        while not self._resume_event.is_set():
            if self._abort_requested:
                return False
            time.sleep(0.01)
        return not self._abort_requested

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

    def _plan_exec_joints(self, joints_rad: List[float]) -> bool:
        try:
            self._pc.set_start_state_to_current_state()

            gs = RobotState(self._moveit.get_robot_model())
            gs.set_joint_group_positions(self.group_name, joints_rad)
            gs.update()
            self._pc.set_goal_state(robot_state=gs)

            params = PlanRequestParameters(self._moveit, "")
            params.planning_pipeline = self._default_pipeline
            params.planning_time = self.planning_time
            params.planning_attempts = self.planning_attempts
            params.max_velocity_scaling_factor = self.max_vel
            params.max_acceleration_scaling_factor = self.max_acc

            result = self._pc.plan(single_plan_parameters=params)
            if not result:
                self.get_logger().error("Planning failed.")
                return False
            return self._execute_trajectory(result.trajectory)
        except Exception as e:
            self.get_logger().error(f"Joint plan/exec failed: {e}")
            return False

    def _read_csv_col(self, path: str, col: str) -> List[Any]:
        out = []
        with open(path, "r", newline="") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                out.append(r[col])
        return out

    def _compute_trial_metrics(self) -> Dict[str, Any]:
        started = int(self.started_flag)

        time_to_success = float("nan")
        if self.outcome == "success" and started == 1 and math.isfinite(self.t_success_s) and math.isfinite(self.t_start_s):
            time_to_success = float(self.t_success_s - self.t_start_s)

        false_start = 0
        gt_action_at_start = -1
        if started == 1 and self.model is not None and math.isfinite(self.t_start_s):
            i = _nearest_index(self.model["center_time_s"], self.t_start_s)
            gt_action_at_start = int(self.model["gt_action"][i])
            false_start = 1 if gt_action_at_start == 0 else 0

        cv_infeas_start = 0
        f_cv_at_start = -1
        if started == 1 and math.isfinite(self.t_start_s) and os.path.exists(self.cv_path):
            t_list = [float(x) for x in self._read_csv_col(self.cv_path, "t_s")]
            f_list = [int(float(x)) for x in self._read_csv_col(self.cv_path, "f_cv")]
            if len(t_list) > 0:
                j = int(np.argmin(np.abs(np.array(t_list) - float(self.t_start_s))))
                f_cv_at_start = int(f_list[j])
                cv_infeas_start = 1 if f_cv_at_start == 0 else 0

        flaps = 0
        if self.model is not None:
            cs = [str(x).strip().upper() for x in self.model["commit_state"]]
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
                self.trial_id, self.variant, m["started"], self.outcome,
                self.t_start_s, self.t_success_s, m["time_to_success"],
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

    def _mark_started(self):
        self.started_flag = 1
        self.t_start_s = self._t_s()
        self.sup_state = "EXECUTE"
        self.get_logger().info(f"[START] t={self.t_start_s:.3f}s f_cv={self.f_cv} closest={self.closest_m:.3f}m")

    def _finish(self, outcome: str):
        self.outcome = outcome
        if outcome == "success":
            self.t_success_s = self._t_s()
        else:
            self.t_success_s = float("nan")

        self._append_csv(
            self.attempts_path,
            [self.trial_id, self.variant, int(self.started_flag), self.t_start_s, self.t_success_s, self.outcome],
        )

        try:
            self._write_metrics_files()
        except Exception as e:
            self.get_logger().error(f"Metrics write failed: {e}")

        self.get_logger().info(
            f"[FINISH] outcome={self.outcome} started={self.started_flag} "
            f"log_dir={self.log_dir} global_summary={self.global_summary_csv}"
        )

    def _wait_until(self, t_target: float):
        while rclpy.ok() and self._t_s() < t_target:
            time.sleep(0.005)

    def _hold_commit_gate_before_step(self) -> bool:
        if self._hold_commit_ignore_cv:
            return True
        if (not self._hold_commit_tripped) and self.f_cv == 0:
            self._hold_commit_tripped = True
            self.sup_state = "HOLD"
            self.get_logger().info("[hold_commit] CV breach during APPROACH -> HOLD 5s then continue (CV ignored)")
            time.sleep(5.0)
            self._hold_commit_ignore_cv = True
            if self.started_flag == 1:
                self.sup_state = "EXECUTE"
            return True
        return True

    def _run(self):
        try:
            self.sup_state = "HOLD"
            self._phase = "IDLE"

            needs_commit = self.variant in ("commit_only", "hold_commit", "hac")
            if needs_commit and self.t_commit_s is None:
                self.get_logger().error("This variant needs pred_csv with commit_event==1, but none was found.")
                self.sup_state = "ABORT"
                self._finish("abort")
                return

            if self.variant == "feas_only":
                while rclpy.ok() and not self._stable_feasible():
                    self.sup_state = "HOLD"
                    time.sleep(0.01)
                self._mark_started()
            else:
                self.sup_state = "HOLD"
                self.get_logger().info(f"Waiting for commit time t_commit={self.t_commit_s:.3f}s ...")
                self._wait_until(float(self.t_commit_s))

                if self.variant == "commit_only":
                    self._mark_started()

                elif self.variant == "hold_commit":
                    self._mark_started()

                elif self.variant == "hac":
                    if self._stable_feasible():
                        self._mark_started()
                    else:
                        self.sup_state = "ASSIST"
                        while rclpy.ok() and not self._stable_feasible():
                            time.sleep(0.01)
                        self._mark_started()

                else:
                    self.get_logger().error(f"Unknown variant: {self.variant}")
                    self.sup_state = "ABORT"
                    self._finish("abort")
                    return

            if self.started_flag != 1:
                self.sup_state = "BLOCKED"
                self._finish("blocked")
                return

            self._phase = "PREP"
            self.get_logger().info("Moving to START pose (gate disabled)...")
            if not self._plan_exec_joints(self.start_joints_rad):
                self.sup_state = "ABORT"
                self._finish("abort")
                return
            time.sleep(max(0.0, self.pause_sec))

            self._phase = "APPROACH"
            self.get_logger().info("Traveling START -> WAVE_READY ...")

            targets: List[List[float]] = []
            for k in range(1, self.approach_steps + 1):
                t = k / float(self.approach_steps)
                q = [_lerp(self.start_joints_rad[i], self.wave_ready_joints_rad[i], t) for i in range(6)]
                targets.append(q)

            for q in targets:
                if self._abort_requested:
                    self.sup_state = "ABORT"
                    self._finish("abort")
                    return

                if self.variant == "feas_only":
                    if self.f_cv == 0:
                        self._request_abort("CV breach during APPROACH")
                        self.sup_state = "ABORT"
                        self._finish("abort")
                        return

                if self.variant == "hac":
                    if not self._wait_if_paused():
                        self.sup_state = "ABORT"
                        self._finish("abort")
                        return

                if self.variant == "hold_commit":
                    if not self._hold_commit_gate_before_step():
                        self.sup_state = "ABORT"
                        self._finish("abort")
                        return

                if not self._plan_exec_joints(q):
                    self.sup_state = "ABORT"
                    self._finish("abort")
                    return

                time.sleep(max(0.0, self.pause_sec))

            self._phase = "WAVE"
            self.sup_state = "EXECUTE"
            self.get_logger().info("Reached WAVE_READY. Starting wave.")

            base = list(self.wave_ready_joints_rad)
            min_pose = list(base)
            max_pose = list(base)
            min_pose[self.wave_joint_index] = self.wave_min_rad
            max_pose[self.wave_joint_index] = self.wave_max_rad

            for _ in range(self.wave_cycles):
                if not self._plan_exec_joints(min_pose):
                    self.sup_state = "ABORT"
                    self._finish("abort")
                    return
                time.sleep(max(0.0, self.pause_sec))

                if not self._plan_exec_joints(max_pose):
                    self.sup_state = "ABORT"
                    self._finish("abort")
                    return
                time.sleep(max(0.0, self.pause_sec))

            if len(self.rest_joints_rad) == 6:
                self.get_logger().info("Returning to REST pose ...")
                self._plan_exec_joints(self.rest_joints_rad)

            self.sup_state = "DONE"
            self._finish("success")

        except Exception as e:
            self.get_logger().error(f"Wave supervisor exception: {e}")
            self.sup_state = "ABORT"
            self._finish("abort")


def main(args=None):
    rclpy.init(args=args)
    node = WaveHACSupervisor()
    try:
        while rclpy.ok() and node._thread.is_alive():
            rclpy.spin_once(node, timeout_sec=0.1)
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