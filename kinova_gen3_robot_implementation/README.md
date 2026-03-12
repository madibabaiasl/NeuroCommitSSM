# Kinova Gen3 Robot Implementation

ROS 2 task supervisors and perception nodes for a **Kinova Gen3 6-DoF** arm with a **Robotiq 2F-85** gripper. Each task implements one of four experimental obstacle-aware execution variants: `commit_only`, `feas_only`, `hold_commit`, and `hac` (Hold Assist Commit).

---

## Setup

### 1. Install ROS 2 Jazzy

Follow the official installation guide:
[https://docs.ros.org/en/jazzy/Installation.html](https://docs.ros.org/en/jazzy/Installation.html)

After installation, source the workspace in every new terminal:

```bash
source /opt/ros/jazzy/setup.bash
```

### 2. Install MoveIt 2

```bash
sudo apt install ros-jazzy-moveit
```

Also install the Kinova MoveIt config package:

```bash
sudo apt install ros-jazzy-kinova-gen3-6dof-robotiq-2f-85-moveit-config
```

> If the package is not available via apt, build it from source:
> [https://github.com/Kinovarobotics/ros2_kortex](https://github.com/Kinovarobotics/ros2_kortex)

### 3. Install Python Dependencies

```bash
pip install numpy scipy opencv-python requests ultralytics
```

- `ultralytics` is only required for T2 (Bottle Pick).
- A [Roboflow](https://roboflow.com/) API key is required for T1, T3, and T4.

### 4. Launch the Kinova Gen3 Arm

Power on the arm and connect it via ethernet. Then bring up the robot driver and MoveIt:

```bash
# Terminal 1 — Robot driver
ros2 launch kortex_bringup gen3.launch.py \
  robot_ip:=192.168.1.10 \
  gripper:=robotiq_2f_85 \
  dof:=6

# Terminal 2 — MoveIt
ros2 launch kinova_gen3_6dof_robotiq_2f_85_moveit_config robot.launch.py
```

> Adjust `robot_ip` to match your network configuration. The driver publishes `/joint_states`, the gripper action server (`/robotiq_gripper_controller/gripper_cmd`), and the planning scene service (`/apply_planning_scene`).

### 5. Launch the Depth Camera

For an Intel RealSense camera:

```bash
sudo apt install ros-jazzy-realsense2-camera
ros2 launch realsense2_camera rs_launch.py \
  enable_depth:=true \
  enable_color:=true \
  align_depth.enable:=true
```

This publishes the required topics:
- `/camera/color/image_raw`
- `/camera/depth_registered/image_rect` (or `/camera/depth/image_raw`)
- `/camera/color/camera_info`

### 6. Camera-to-Robot Calibration (TF)

The TF tree must include a `base_link` → `camera_color_frame` transform. If the camera is rigidly mounted, publish a static transform:

```bash
ros2 run tf2_ros static_transform_publisher \
  --x 0.0 --y 0.0 --z 0.0 \
  --roll 0.0 --pitch 0.0 --yaw 0.0 \
  --frame-id base_link --child-frame-id camera_color_frame
```

> Replace the values with your measured extrinsic calibration. For hand-eye calibration, see the [MoveIt calibration tutorial](https://moveit.picknik.ai/main/doc/examples/hand_eye_calibration/hand_eye_calibration_tutorial.html).

---

## Running a Task

Once the robot arm, MoveIt, and camera are all running, launch task nodes in additional terminals.

All nodes are registered under the `bottle_grasping` ROS 2 package:

```bash
ros2 run bottle_grasping <node_name> --ros-args -p <param>:=<value>
```

---

## Tasks

### T1 — Clock Pick (`T1_clock/`)

Pick up a digital alarm clock by its corner.

**Nodes (run in two terminals):**

```bash
# Terminal 1 — Perception (must start first)
ros2 run bottle_grasping clock_pose_node_metrics \
  --rf_api_key <ROBOFLOW_KEY> --rf_project clock-geqo0 --rf_version 1

# Terminal 2 — Supervisor
ros2 run bottle_grasping clock_pick_supervisor_metrics --ros-args \
  -p variant:=commit_only \
  -p trial_id:=1 \
  -p log_dir:=/tmp/clock_logs \
  -p global_summary_csv:=/tmp/clock_metrics_summary.csv
```

**Pipeline:** Roboflow detects clock → pose node publishes marker → supervisor grasps corner → releases → retreats.

---

### T2 — Bottle Pick (`T2_bottle/`)

Pick up a bottle and place it at a target location. Perception (YOLOv8) is built into the supervisor — no separate pose node needed.

**Node (single terminal):**

```bash
ros2 run bottle_grasping bottle_pick_supervisor_metrics --ros-args \
  -p variant:=commit_only \
  -p trial_id:=1 \
  -p log_dir:=/tmp/bottle_logs \
  -p global_summary_csv:=/tmp/bottle_metrics_summary.csv
```

> Requires `yolov8n-seg.pt` model file (auto-downloaded by `ultralytics` on first run).

**Pipeline:** YOLO detects bottle (COCO class 39) → standoff approach → cartesian LIN grasp → place → release.

---

### T3 — Fan Button Press (`T3_fan/`)

Press the white power button on a fan.

**Nodes (run in two terminals):**

```bash
# Terminal 1 — Perception (must start first)
ros2 run bottle_grasping fan_pose_buttons_node_v2 \
  --rf_api_key <ROBOFLOW_KEY> --rf_project fan-buttons --rf_version 1

# Terminal 2 — Supervisor
ros2 run bottle_grasping fan_press_supervisor --ros-args \
  -p variant:=feas_only \
  -p trial_id:=1
```

**Pipeline:** Roboflow detects fan + buttons → pose node publishes 3D button positions & CV feasibility → supervisor presses white button with force-limited motion (max 25 N).

---

### T4 — Plant Move (`T4_plant/`)

Push a potted plant sideways using an L-shaped sweep.

**Nodes (run in two terminals):**

```bash
# Terminal 1 — Perception (must start first)
ros2 run bottle_grasping plant_pose_node_metrics --mode detect

# Terminal 2 — Supervisor
ros2 run bottle_grasping plant_move_supervisor_metrics --ros-args \
  -p variant:=commit_only \
  -p trial_id:=1 \
  -p pred_csv:=/path/to/pred.csv \
  -p log_dir:=/tmp/plant_logs \
  -p global_summary_csv:=/tmp/plant_metrics_summary.csv
```

> The pose node needs a Roboflow API key (project `plant-fjnqj`). Set it via `--rf_api_key`.

**Pipeline:** Roboflow detects plant → pose node estimates pot center → supervisor moves end-effector 5 in. left of plant → advances forward → sweeps right 8 in. → returns to rest.

---

### T5 — Wave (`T5_wave/`)

Wave the robot arm. Uses raw depth for proximity-based feasibility — no object detection needed.

**Node (single terminal):**

```bash
ros2 run bottle_grasping wave_supervisor_final --ros-args \
  -p variant:=commit_only \
  -p trial_id:=1 \
  -p pred_csv:=/path/to/pred.csv \
  -p log_dir:=/tmp/wave_logs \
  -p global_summary_csv:=/tmp/wave_metrics_summary.csv
```

**Pipeline:** Depth proximity gate (`f_cv = 1` when nearest obstacle > 0.65 m) → move to wave-ready pose → oscillate joint 3 for 2 cycles.

---

### Trest — Hold to Rest (`Trest_hold/`)

Utility supervisor that follows a CSV timeline and moves the arm to rest whenever the commit state transitions to `HOLD`. No perception or obstacle logic.

**Node (single terminal):**

```bash
ros2 run bottle_grasping hold_to_rest_supervisor --ros-args \
  -p pred_csv:=/path/to/pred.csv
```

**Pipeline:** Polls CSV at 50 ms intervals → on `HOLD` transition → plans and executes motion to rest joint configuration.
