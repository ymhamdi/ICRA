"""
==============================================================================
  ALVE-XXX ROBOT ARM  –  Vision Pick & Place Pipeline
  ====================================================
  Dependencies: pip install pybullet numpy opencv-python

  Features:
    1. Synthetic overhead camera rendering
    2. OpenCV HSV color detection & centroid extraction
    3. Pixel-to-World coordinate unprojection using Depth Buffer
    4. Inverse Kinematics (IK) mapping to joint space
    5. Automated Keyframe Sequence generation & execution
==============================================================================
"""

import os
import sys
import time
import math
import tempfile
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Tuple

try:
    import pybullet as p
    import pybullet_data
    import numpy as np
    import cv2
except ImportError as e:
    sys.exit(f"[ERROR] Missing dependency: {e}. Run: pip install pybullet numpy opencv-python")

# =============================================================================
# ── PATH CONFIGURATION ────────────────────────────────────────────────────────
# =============================================================================
SCRIPT_DIR   = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR / "alve-xxx-robot-arm-urdf-main"
URDF_SRC     = PACKAGE_ROOT / "urdf" / "ALVE-XXX ROBOT ARM urdf FINAL.urdf"
MESHES_DIR   = PACKAGE_ROOT / "meshes"
PATCHED_URDF_NAME = "alve_arm_patched.urdf"

# =============================================================================
# ── DATA CLASSES ─────────────────────────────────────────────────────────────
# =============================================================================
@dataclass
class Slide:
    joint_angles : List[float]
    gripper      : str   = "keep"   # "open" | "close" | "keep"
    duration     : float = 2.0
    label        : str   = "slide"
    pause_after  : float = 0.3

# =============================================================================
# ── URDF PATCHER ──────────────────────────────────────────────────────────────
# =============================================================================
def patch_urdf(src_path: Path, meshes_dir: Path, out_dir: Path) -> Path:
    out_path = out_dir / PATCHED_URDF_NAME
    tree = ET.parse(src_path)
    root = tree.getroot()
    for mesh_elem in root.iter("mesh"):
        filename_attr = mesh_elem.get("filename", "")
        if filename_attr.startswith("package://"):
            basename = Path(filename_attr).name
            abs_path = (meshes_dir / basename).resolve()
            mesh_elem.set("filename", abs_path.as_posix())
    tree.write(str(out_path), encoding="utf-8", xml_declaration=True)
    return out_path

# =============================================================================
# ── ROBOT ARM VISION & IK CLASS ───────────────────────────────────────────────
# =============================================================================
class VisionPickAndPlaceSim:
    ARM_JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
    GRIPPER_PRIMARY = "r_grip_joint"
    GRIPPER_MIMIC = {
        "j_r_EE": 1.0, "j_l_grip": -1.0, "j_l_EE": 1.0,
        "joint_grip_l1": 1.0, "joint_grip_l2": -1.0
    }
    
    GRIPPER_OPEN_POS  = 0.0
    GRIPPER_CLOSE_POS = 0.65
    GRIPPER_FORCE   = 10.0
    ARM_FORCE       = 50.0
    ARM_MAX_VEL     = 2.0
    SIM_TIMESTEP    = 1.0 / 240.0
    INTERP_STEPS_PER_SEC = 120

    def __init__(self, gui: bool = True):
        self.gui = gui
        self.robot_id = None
        self._physics_client = None
        self.arm_joint_indices = []
        self.joint_name_to_idx = {}
        self.gripper_mimic_indices = {}
        self._gripper_pos = self.GRIPPER_OPEN_POS
        self.cube_id = None

    def connect(self):
        mode = p.GUI if self.gui else p.DIRECT
        self._physics_client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.SIM_TIMESTEP)
        
        if self.gui:
            p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0.2, 0, 0])
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
            
        p.loadURDF("plane.urdf")

    def load_robot(self, patched_urdf: Path):
        self.robot_id = p.loadURDF(str(patched_urdf), useFixedBase=True)
        
        for i in range(p.getNumJoints(self.robot_id)):
            name = p.getJointInfo(self.robot_id, i)[1].decode()
            self.joint_name_to_idx[name] = i
            
        self.arm_joint_indices = [self.joint_name_to_idx[n] for n in self.ARM_JOINT_NAMES]
        self.gripper_primary_idx = self.joint_name_to_idx[self.GRIPPER_PRIMARY]
        
        for name in self.GRIPPER_MIMIC:
            if name in self.joint_name_to_idx:
                self.gripper_mimic_indices[name] = self.joint_name_to_idx[name]
                
        # Move home and let physics settle
        for _ in range(120): p.stepSimulation()
        
    def spawn_target_object(self):
        """Spawn a red cube on the floor to be detected"""
        print("\n[SCENE] Spawning target red cube...")
        # Randomize spawn location slightly for dynamic proof
        spawn_x = np.random.uniform(0.3, 0.45)
        spawn_y = np.random.uniform(-0.15, 0.15)
        self.cube_id = p.loadURDF("cube_small.urdf", [spawn_x, spawn_y, 0.025], globalScaling=0.5)
        p.changeVisualShape(self.cube_id, -1, rgbaColor=[1, 0, 0, 1])
        # Let it drop and settle
        for _ in range(100): p.stepSimulation()

    # ── Vision & OpenCV ──────────────────────────────────────────────────────
    def get_camera_frame(self):
        """Setup synthetic overhead camera"""
        print("[VISION] Capturing synthetic camera frame...")
        width, height = 640, 480
        # Position camera overhead looking down at the table/floor
        cam_eye = [0.35, 0.0, 0.8]
        cam_target = [0.35, 0.0, 0.0]
        cam_up = [1, 0, 0] # Pointing along +X
        
        view_matrix = p.computeViewMatrix(cam_eye, cam_target, cam_up)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60.0, aspect=float(width)/height, nearVal=0.1, farVal=2.0)
        
        img_arr = p.getCameraImage(width, height, view_matrix, proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        
        rgb = np.reshape(img_arr[2], (height, width, 4))[:, :, :3]
        depth = np.reshape(img_arr[3], (height, width))
        return rgb, depth, view_matrix, proj_matrix, width, height

    def detect_red_object(self, rgb) -> Optional[Tuple[int, int]]:
        """Find the pixel coordinates (cx, cy) of the red cube using OpenCV"""
        print("[VISION] Running OpenCV HSV color thresholding for 'Red'...")
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        
        # Red hue wraps around in HSV
        mask1 = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
        mask = mask1 + mask2
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
            
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] == 0: return None
        
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        
        # Save debug image
        cv2.circle(bgr, (cx, cy), 8, (0, 255, 0), -1)
        cv2.putText(bgr, f"Target ({cx},{cy})", (cx+10, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        cv2.imwrite("vision_debug.png", bgr)
        print("  ✓ Target found. Debug frame saved to 'vision_debug.png'")
        return cx, cy

    def pixel_to_world(self, cx, cy, depth_buffer, view_mat, proj_mat, width, height):
        """Unproject 2D pixel to 3D world space coordinate"""
        print("[VISION] Mapping pixel to 3D World Coordinates...")
        depth_val = depth_buffer[cy, cx]
        
        # NDC Coordinates
        ndc_x = (2.0 * cx / width) - 1.0
        ndc_y = 1.0 - (2.0 * cy / height)
        ndc_z = 2.0 * depth_val - 1.0
        clip_coords = np.array([ndc_x, ndc_y, ndc_z, 1.0])
        
        # Note: PyBullet outputs column-major arrays. We reshape to row-major by Fortran order.
        pm = np.array(proj_mat).reshape((4, 4), order='F')
        vm = np.array(view_mat).reshape((4, 4), order='F')
        
        eye_coords = np.linalg.inv(pm) @ clip_coords
        eye_coords /= eye_coords[3]
        
        world_coords = np.linalg.inv(vm) @ eye_coords
        print(f"  ✓ World Coordinate: X={world_coords[0]:.3f}, Y={world_coords[1]:.3f}, Z={world_coords[2]:.3f}")
        return world_coords[:3]

    # ── Inverse Kinematics ───────────────────────────────────────────────────
    def calculate_ik(self, target_pos, target_quat):
        """Compute the 6 joint angles needed to reach target_pos"""
        ee_index = self.joint_name_to_idx["joint_6"]
        
        # Calculate IK for all movable joints
        joint_poses = p.calculateInverseKinematics(
            self.robot_id, ee_index, target_pos, target_quat,
            maxNumIterations=100, residualThreshold=1e-5
        )
        
        # Extract only the 6 arm joints from the IK solution
        movable_joints = [i for i in range(p.getNumJoints(self.robot_id)) if p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED]
        return [joint_poses[movable_joints.index(idx)] for idx in self.arm_joint_indices]

    # ── Low Level Control ────────────────────────────────────────────────────
    def _apply_gripper_raw(self, pos: float):
        p.setJointMotorControl2(self.robot_id, self.gripper_primary_idx, p.POSITION_CONTROL, pos, force=self.GRIPPER_FORCE, maxVelocity=1.5)
        for name, mult in self.GRIPPER_MIMIC.items():
            if name in self.gripper_mimic_indices:
                p.setJointMotorControl2(self.robot_id, self.gripper_mimic_indices[name], p.POSITION_CONTROL, pos * mult, force=self.GRIPPER_FORCE, maxVelocity=1.5)
        self._gripper_pos = pos

    def open_gripper(self):
        for i in range(60):
            t = (i + 1) / 60
            self._apply_gripper_raw(self._gripper_pos + t * (self.GRIPPER_OPEN_POS - self._gripper_pos))
            p.stepSimulation()
            if self.gui: time.sleep(self.SIM_TIMESTEP)

    def close_gripper(self):
        for i in range(60):
            t = (i + 1) / 60
            self._apply_gripper_raw(self._gripper_pos + t * (self.GRIPPER_CLOSE_POS - self._gripper_pos))
            p.stepSimulation()
            if self.gui: time.sleep(self.SIM_TIMESTEP)

    # ── Slide Sequence Execution ─────────────────────────────────────────────
    def _cubic_ease(self, t: float) -> float: return t * t * (3.0 - 2.0 * t)

    def move_to_slide(self, slide: Slide):
        if slide.gripper == "open": self.open_gripper()
        elif slide.gripper == "close": self.close_gripper()
        
        starts = [p.getJointState(self.robot_id, idx)[0] for idx in self.arm_joint_indices]
        total_steps = max(10, int(slide.duration * self.INTERP_STEPS_PER_SEC))
        step_sleep = slide.duration / total_steps

        for step in range(total_steps):
            t = self._cubic_ease((step + 1) / total_steps)
            for idx, start, target in zip(self.arm_joint_indices, starts, slide.joint_angles):
                p.setJointMotorControl2(self.robot_id, idx, p.POSITION_CONTROL, start + t * (target - start), force=self.ARM_FORCE, maxVelocity=self.ARM_MAX_VEL)
            self._apply_gripper_raw(self._gripper_pos)
            p.stepSimulation()
            if self.gui: time.sleep(step_sleep)

        for _ in range(max(10, int(slide.pause_after / self.SIM_TIMESTEP))):
            self._apply_gripper_raw(self._gripper_pos)
            p.stepSimulation()
            if self.gui: time.sleep(self.SIM_TIMESTEP)

    def run_sequence(self, slides: List[Slide]):
        print(f"\n[SEQ] Executing Automated Pick-and-Place ({len(slides)} slides)")
        for i, slide in enumerate(slides, 1):
            print(f"  ▶ [{i}/{len(slides)}] {slide.label}")
            self.move_to_slide(slide)
        print("[SEQ] Complete!\n")

    def disconnect(self):
        if self._physics_client is not None:
            p.disconnect(self._physics_client)

    def __enter__(self): return self
    def __exit__(self, *args): self.disconnect()


# =============================================================================
# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────
# =============================================================================
def main():
    print("\n" + "═"*58)
    print("  ALVE-XXX Vision Pick & Place")
    print("═"*58 + "\n")

    tmp_dir = Path(tempfile.mkdtemp(prefix="alve_arm_"))
    patched = patch_urdf(URDF_SRC, MESHES_DIR, tmp_dir)

    try:
        with VisionPickAndPlaceSim(gui=True) as sim:
            sim.connect()
            sim.load_robot(patched)
            sim.spawn_target_object()

            # --- 1. Vision Phase ---
            rgb, depth, vmat, pmat, w, h = sim.get_camera_frame()
            centroid = sim.detect_red_object(rgb)
            if not centroid:
                raise RuntimeError("Failed to detect the red cube in the camera frame.")
            cx, cy = centroid

            # --- 2. Mapping Phase ---
            obj_pos = sim.pixel_to_world(cx, cy, depth, vmat, pmat, w, h)
            
            # --- 3. Inverse Kinematics Phase ---
            # Gripper length offset so wrist doesn't crash into object
            z_offset_hover = 0.35 
            z_offset_grasp = 0.22 
            
            # Quat to point gripper down (180 deg around Y axis)
            down_quat = p.getQuaternionFromEuler([0, math.pi, 0])

            print("\n[IK] Computing Trajectory...")
            ik_hover = sim.calculate_ik([obj_pos[0], obj_pos[1], obj_pos[2] + z_offset_hover], down_quat)
            ik_grasp = sim.calculate_ik([obj_pos[0], obj_pos[1], obj_pos[2] + z_offset_grasp], down_quat)
            
            drop_loc = [0.0, 0.35, 0.0] # Pre-defined drop-off zone
            ik_drop_hover = sim.calculate_ik([drop_loc[0], drop_loc[1], drop_loc[2] + z_offset_hover], down_quat)
            ik_drop = sim.calculate_ik([drop_loc[0], drop_loc[1], drop_loc[2] + z_offset_grasp + 0.05], down_quat)
            ik_home = [0.0]*6

            # --- 4. Generate & Execute Sequence ---
            sequence = [
                Slide(ik_hover,       gripper="open",  duration=2.0, label="Hover above object", pause_after=0.5),
                Slide(ik_grasp,       gripper="keep",  duration=1.5, label="Descend to grasp",   pause_after=0.2),
                Slide(ik_grasp,       gripper="close", duration=0.5, label="GRASP object",       pause_after=0.5),
                Slide(ik_hover,       gripper="keep",  duration=1.5, label="Lift object",        pause_after=0.2),
                Slide(ik_drop_hover,  gripper="keep",  duration=2.0, label="Move to drop zone",  pause_after=0.2),
                Slide(ik_drop,        gripper="keep",  duration=1.5, label="Descend to drop",    pause_after=0.2),
                Slide(ik_drop,        gripper="open",  duration=0.5, label="RELEASE object",     pause_after=0.5),
                Slide(ik_home,        gripper="keep",  duration=2.0, label="Return to Home",     pause_after=0.5),
            ]
            
            sim.run_sequence(sequence)

            print("[SIM] Sequence finished. Close window to exit.")
            while True:
                p.stepSimulation()
                sim._apply_gripper_raw(sim._gripper_pos)
                time.sleep(sim.SIM_TIMESTEP)

    except KeyboardInterrupt:
        pass
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
