"""
==============================================================================
  ALVE-XXX ROBOT ARM  –  PyBullet Keyframe Sequencer
  ====================================================
  Author : generated for ICRA project
  Target : Python 3.8+  |  pybullet >= 3.2

  HOW TO USE
  ----------
  1.  Install deps:   pip install pybullet numpy
  2.  Run:            python alve_arm_simulation.py
  3.  To add a new "slide", append a Slide(...) entry to MOVEMENT_SEQUENCE
      inside main() – see the "HOW TO ADD NEW SLIDES" comment block.

  JOINT ORDER in every Slide
  --------------------------
    [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6]
     (all in radians, limits ±π/2 ≈ ±1.5708)

  GRIPPER
  -------
    open_gripper()  / close_gripper()  are called as part of any slide.
    The primary driver joint is r_grip_joint; the five mimic joints are
    updated manually each physics step (PyBullet ignores <mimic> tags).
==============================================================================
"""

import os
import sys
import time
import math
import shutil
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple

# ── dependency check ─────────────────────────────────────────────────────────
try:
    import pybullet as p
    import pybullet_data
except ImportError:
    sys.exit(
        "[ERROR] pybullet not found.  Run:  pip install pybullet\n"
        "        then re-run this script."
    )

try:
    import numpy as np
except ImportError:
    sys.exit(
        "[ERROR] numpy not found.  Run:  pip install numpy\n"
        "        then re-run this script."
    )

# =============================================================================
# ── PATH CONFIGURATION ────────────────────────────────────────────────────────
# =============================================================================

# Root of the URDF package  (adjust if you move the folder)
SCRIPT_DIR   = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR / "alve-xxx-robot-arm-urdf-main"
URDF_SRC     = PACKAGE_ROOT / "urdf" / "ALVE-XXX ROBOT ARM urdf FINAL.urdf"
MESHES_DIR   = PACKAGE_ROOT / "meshes"

# PyBullet cannot resolve  package://...  URIs, so we write a patched copy
# into a temp file that uses absolute paths.
PATCHED_URDF_NAME = "alve_arm_patched.urdf"


# =============================================================================
# ── DATA CLASSES ─────────────────────────────────────────────────────────────
# =============================================================================

@dataclass
class Slide:
    """
    One keyframe in the movement sequence.

    Parameters
    ----------
    joint_angles : list[float]
        Target angles (radians) for the six arm joints in order:
        [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6]
    gripper : str
        "open"  → fingers fully open
        "close" → fingers fully closed / grasping
        "keep"  → maintain the current gripper state
    duration : float
        Time (seconds) to reach this pose from the previous one.
    label : str
        Human-readable name shown in the terminal while executing.
    pause_after : float
        Extra wait time (seconds) after arriving at the pose.
    """
    joint_angles : List[float]
    gripper      : str   = "keep"   # "open" | "close" | "keep"
    duration     : float = 2.0
    label        : str   = "slide"
    pause_after  : float = 0.3

    def __post_init__(self):
        if len(self.joint_angles) != 6:
            raise ValueError(
                f"Slide '{self.label}': joint_angles must have exactly 6 values, "
                f"got {len(self.joint_angles)}."
            )
        valid = {"open", "close", "keep"}
        if self.gripper not in valid:
            raise ValueError(
                f"Slide '{self.label}': gripper must be one of {valid}, "
                f"got '{self.gripper}'."
            )


# =============================================================================
# ── URDF PATCHER ──────────────────────────────────────────────────────────────
# =============================================================================

def patch_urdf(src_path: Path, meshes_dir: Path, out_dir: Path) -> Path:
    """
    Rewrite every  package://...  mesh URI in the URDF with an absolute file
    path that PyBullet can open directly.

    Returns the path to the patched URDF file.
    """
    out_path = out_dir / PATCHED_URDF_NAME

    tree = ET.parse(src_path)
    root = tree.getroot()

    replaced = 0
    for mesh_elem in root.iter("mesh"):
        filename_attr = mesh_elem.get("filename", "")
        if filename_attr.startswith("package://"):
            # Strip the  package://<pkg-name>/meshes/  prefix, keep basename
            basename = Path(filename_attr).name
            abs_path = (meshes_dir / basename).resolve()
            if not abs_path.exists():
                print(f"  [WARN] Mesh file not found: {abs_path}")
            # PyBullet needs forward slashes even on Windows
            mesh_elem.set("filename", abs_path.as_posix())
            replaced += 1

    tree.write(str(out_path), encoding="utf-8", xml_declaration=True)
    print(f"[INFO] URDF patched: {replaced} mesh paths rewritten → {out_path}")
    return out_path


# =============================================================================
# ── ENVIRONMENT VALIDATION ────────────────────────────────────────────────────
# =============================================================================

def validate_environment(package_root: Path, urdf_src: Path, meshes_dir: Path):
    """
    Check that all required files exist before starting the simulation.
    Raises FileNotFoundError with a clear message on the first missing item.
    """
    print("\n[CHECK] Validating environment …")

    if not package_root.exists():
        raise FileNotFoundError(
            f"Package folder not found:\n  {package_root}\n"
            "Expected:  alve-xxx-robot-arm-urdf-main/  next to this script."
        )

    if not urdf_src.exists():
        raise FileNotFoundError(
            f"URDF file not found:\n  {urdf_src}"
        )

    if not meshes_dir.exists():
        raise FileNotFoundError(
            f"Meshes folder not found:\n  {meshes_dir}"
        )

    expected_meshes = [
        "base_link.STL", "link_1.STL", "link_2.STL", "link_3.STL",
        "link_4.STL",    "link_5.STL", "link_6.STL", "servo.STL",
        "r_grip.STL",    "r_EE.STL",   "l_grip.STL", "l_EE.STL",
        "grip_l1.STL",   "grip_l2.STL",
    ]
    missing = []
    for mesh in expected_meshes:
        if not (meshes_dir / mesh).exists():
            missing.append(mesh)

    if missing:
        raise FileNotFoundError(
            f"Missing mesh files in {meshes_dir}:\n  " +
            "\n  ".join(missing)
        )

    print(f"  ✓  Package root  : {package_root}")
    print(f"  ✓  URDF          : {urdf_src.name}")
    print(f"  ✓  All {len(expected_meshes)} mesh files present")
    print("[CHECK] Environment OK.\n")


# =============================================================================
# ── ROBOT ARM SIMULATION CLASS ────────────────────────────────────────────────
# =============================================================================

class AlveArmSimulation:
    """
    Manages the PyBullet physics world, robot loading, joint control,
    and keyframe-based movement sequencing for the ALVE-XXX robot arm.
    """

    # ── Joint / gripper constants ────────────────────────────────────────────

    # Names of the six controllable arm joints (must match URDF joint names)
    ARM_JOINT_NAMES: List[str] = [
        "joint_1", "joint_2", "joint_3",
        "joint_4", "joint_5", "joint_6",
    ]

    # The PRIMARY gripper joint (drives open/close).
    # All others are mimic joints updated manually.
    GRIPPER_PRIMARY = "r_grip_joint"

    # mimic joint → multiplier relative to r_grip_joint
    GRIPPER_MIMIC: Dict[str, float] = {
        "j_r_EE"       :  1.0,   # right fingertip extension
        "j_l_grip"     : -1.0,   # left finger (opposite direction)
        "j_l_EE"       :  1.0,   # left fingertip extension
        "joint_grip_l1":  1.0,   # left knuckle link 1
        "joint_grip_l2": -1.0,   # left knuckle link 2
    }

    # Gripper limits (from URDF <limit> tags on r_grip_joint)
    GRIPPER_OPEN_POS  = 0.0    # fingers fully open  (neutral / spread)
    GRIPPER_CLOSE_POS = 0.65   # fingers closed / grasping (< upper 0.75)

    # Control tuning
    GRIPPER_FORCE   = 5.0      # N·m – enough to hold without instability
    ARM_FORCE       = 50.0     # N·m per arm joint
    ARM_MAX_VEL     = 2.0      # rad/s maximum joint velocity

    # Physics
    SIM_TIMESTEP    = 1.0 / 240.0
    INTERP_STEPS_PER_SEC = 120   # how many control updates per second

    def __init__(self, gui: bool = True):
        self.gui      = gui
        self.robot_id : Optional[int] = None
        self._physics_client: Optional[int] = None

        # Populated after URDF load
        self.arm_joint_indices    : List[int]        = []
        self.gripper_primary_idx  : int              = -1
        self.gripper_mimic_indices: Dict[str, int]   = {}
        self.joint_name_to_idx    : Dict[str, int]   = {}
        self.joint_limits         : Dict[int, Tuple[float, float]] = {}

        # Current gripper state (tracks last commanded position)
        self._gripper_pos: float = self.GRIPPER_OPEN_POS

    # ── Initialise PyBullet ──────────────────────────────────────────────────

    def connect(self):
        """Start the PyBullet server and configure the scene."""
        mode = p.GUI if self.gui else p.DIRECT
        self._physics_client = p.connect(mode)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.SIM_TIMESTEP)

        if self.gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=0.7,
                cameraYaw=45,
                cameraPitch=-25,
                cameraTargetPosition=[0, 0, 0.25],
            )
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

        # Ground plane
        p.loadURDF("plane.urdf")
        print("[SIM] PyBullet connected and scene initialised.")

    # ── Load & introspect the robot ──────────────────────────────────────────

    def load_robot(self, patched_urdf: Path):
        """
        Load the ALVE-XXX URDF and map every joint by name.
        """
        print(f"[SIM] Loading robot from:\n      {patched_urdf}")

        self.robot_id = p.loadURDF(
            str(patched_urdf),
            basePosition=[0, 0, 0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION,
        )

        num_joints = p.getNumJoints(self.robot_id)
        print(f"[SIM] Robot loaded (id={self.robot_id}).  "
              f"Total joints found: {num_joints}")
        print(f"\n{'─'*58}")
        print(f"  {'#':>3}  {'Name':<22}  {'Type':<10}  {'Limits'}")
        print(f"{'─'*58}")

        type_names = {
            p.JOINT_REVOLUTE:  "REVOLUTE",
            p.JOINT_PRISMATIC: "PRISMATIC",
            p.JOINT_SPHERICAL: "SPHERICAL",
            p.JOINT_PLANAR:    "PLANAR",
            p.JOINT_FIXED:     "FIXED",
        }

        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i)
            jname  = info[1].decode()
            jtype  = info[2]
            lo, hi = info[8], info[9]

            self.joint_name_to_idx[jname] = i
            if jtype != p.JOINT_FIXED:
                self.joint_limits[i] = (lo, hi)

            limit_str = f"[{lo:+.3f}, {hi:+.3f}]" if jtype != p.JOINT_FIXED else "fixed"
            print(f"  {i:>3}  {jname:<22}  {type_names.get(jtype,'?'):<10}  {limit_str}")

        print(f"{'─'*58}\n")

        # Map arm joints
        missing = []
        for name in self.ARM_JOINT_NAMES:
            if name not in self.joint_name_to_idx:
                missing.append(name)
            else:
                self.arm_joint_indices.append(self.joint_name_to_idx[name])

        if missing:
            raise RuntimeError(
                f"[ERROR] Could not find arm joints in URDF: {missing}\n"
                "        Check that joint names match ARM_JOINT_NAMES."
            )

        # Map gripper primary
        if self.GRIPPER_PRIMARY not in self.joint_name_to_idx:
            raise RuntimeError(
                f"[ERROR] Primary gripper joint '{self.GRIPPER_PRIMARY}' not found."
            )
        self.gripper_primary_idx = self.joint_name_to_idx[self.GRIPPER_PRIMARY]

        # Map gripper mimic joints
        for name in self.GRIPPER_MIMIC:
            if name in self.joint_name_to_idx:
                self.gripper_mimic_indices[name] = self.joint_name_to_idx[name]
            else:
                print(f"  [WARN] Mimic joint '{name}' not found – skipping.")

        print(f"[SIM] Arm joint indices    : {self.arm_joint_indices}")
        print(f"[SIM] Gripper primary idx  : {self.gripper_primary_idx}")
        print(f"[SIM] Gripper mimic indices: {list(self.gripper_mimic_indices.values())}")

        # Reset all joints to zero (home pose)
        for idx in self.arm_joint_indices:
            p.resetJointState(self.robot_id, idx, 0.0)
        self._apply_gripper_raw(self.GRIPPER_OPEN_POS)

        # Warm up a few frames so physics settles
        for _ in range(120):
            p.stepSimulation()

        print("[SIM] Robot ready.\n")

    # ── Low-level gripper control ────────────────────────────────────────────

    def _apply_gripper_raw(self, pos: float):
        """
        Drive the primary gripper joint AND all mimic joints to a given
        position.  Call this every physics step during transitions.
        """
        # Primary joint
        p.setJointMotorControl2(
            self.robot_id,
            self.gripper_primary_idx,
            controlMode=p.POSITION_CONTROL,
            targetPosition=pos,
            force=self.GRIPPER_FORCE,
            maxVelocity=1.5,
        )
        # Mimic joints – update each with its multiplier
        for name, mult in self.GRIPPER_MIMIC.items():
            idx = self.gripper_mimic_indices.get(name)
            if idx is not None:
                p.setJointMotorControl2(
                    self.robot_id,
                    idx,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=pos * mult,
                    force=self.GRIPPER_FORCE,
                    maxVelocity=1.5,
                )
        self._gripper_pos = pos

    def open_gripper(self, steps: int = 120):
        """
        Fully open the parallel gripper over `steps` physics steps.
        Fingers spread apart; safe to call before placing or releasing.
        """
        print("  [GRIP] Opening gripper …")
        target = self.GRIPPER_OPEN_POS
        start  = self._gripper_pos
        for i in range(steps):
            t   = (i + 1) / steps
            pos = start + t * (target - start)
            self._apply_gripper_raw(pos)
            p.stepSimulation()
            if self.gui:
                time.sleep(self.SIM_TIMESTEP)
        print("  [GRIP] Gripper open.")

    def close_gripper(self, steps: int = 180):
        """
        Close the parallel gripper over `steps` physics steps.
        The joint stops at GRIPPER_CLOSE_POS to avoid force-spiking into a
        rigid object.  Increase GRIPPER_FORCE if the object slips.
        """
        print("  [GRIP] Closing gripper …")
        target = self.GRIPPER_CLOSE_POS
        start  = self._gripper_pos
        for i in range(steps):
            t   = (i + 1) / steps
            pos = start + t * (target - start)
            self._apply_gripper_raw(pos)
            p.stepSimulation()
            if self.gui:
                time.sleep(self.SIM_TIMESTEP)
        print("  [GRIP] Gripper closed.")

    # ── Arm interpolation ────────────────────────────────────────────────────

    def _clamp_angle(self, joint_idx: int, angle: float) -> float:
        """Clamp target angle to the joint's URDF limits."""
        lo, hi = self.joint_limits.get(joint_idx, (-math.pi, math.pi))
        clamped = max(lo, min(hi, angle))
        if abs(clamped - angle) > 1e-4:
            info  = p.getJointInfo(self.robot_id, joint_idx)
            jname = info[1].decode()
            print(f"    [WARN] '{jname}' target {angle:.4f} rad clamped to "
                  f"[{lo:.4f}, {hi:.4f}].")
        return clamped

    def get_current_arm_angles(self) -> List[float]:
        """Return the current joint positions for the six arm joints."""
        return [
            p.getJointState(self.robot_id, idx)[0]
            for idx in self.arm_joint_indices
        ]

    def _cubic_ease(self, t: float) -> float:
        """
        Smoothstep (cubic Hermite) easing: slow-start → fast-mid → slow-end.
        Input t ∈ [0, 1], output ∈ [0, 1].
        """
        return t * t * (3.0 - 2.0 * t)

    def move_to_slide(self, slide: Slide):
        """
        Interpolate all six arm joints from their current positions to the
        target angles defined in `slide`, over `slide.duration` seconds.

        The gripper open/close command in slide.gripper is executed *before*
        the arm motion begins (so you can open before approaching and close
        after).
        """
        # ── handle gripper intent ────────────────────────────────────────────
        if slide.gripper == "open":
            self.open_gripper()
        elif slide.gripper == "close":
            self.close_gripper()
        # "keep" → do nothing to gripper

        # ── validate & clamp target angles ──────────────────────────────────
        targets = []
        for i, (idx, angle) in enumerate(
            zip(self.arm_joint_indices, slide.joint_angles)
        ):
            targets.append(self._clamp_angle(idx, angle))

        starts = self.get_current_arm_angles()

        # Decide how many interpolation steps to use
        total_steps = max(10, int(slide.duration * self.INTERP_STEPS_PER_SEC))

        print(f"  [ARM] '{slide.label}'  |  {slide.duration:.1f}s  "
              f"({total_steps} steps)")
        print(f"        targets: {['%+.3f' % a for a in targets]}")

        step_sleep = slide.duration / total_steps   # wall-clock s per step

        for step in range(total_steps):
            t_raw = (step + 1) / total_steps
            t     = self._cubic_ease(t_raw)

            for i, (idx, start, target) in enumerate(
                zip(self.arm_joint_indices, starts, targets)
            ):
                interp = start + t * (target - start)
                p.setJointMotorControl2(
                    self.robot_id,
                    idx,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=interp,
                    force=self.ARM_FORCE,
                    maxVelocity=self.ARM_MAX_VEL,
                )

            # Keep gripper mimic joints in sync every step
            self._apply_gripper_raw(self._gripper_pos)

            p.stepSimulation()
            if self.gui:
                time.sleep(step_sleep)

        # Settle physics after arriving
        settle_steps = int(slide.pause_after / self.SIM_TIMESTEP)
        for _ in range(max(10, settle_steps)):
            self._apply_gripper_raw(self._gripper_pos)
            p.stepSimulation()
            if self.gui:
                time.sleep(self.SIM_TIMESTEP)

        print(f"        → done.\n")

    # ── Sequence runner ──────────────────────────────────────────────────────

    def run_sequence(self, slides: List[Slide]):
        """
        Execute a list of Slide keyframes in order.

        To add new slides: append Slide(...) objects to the list passed in.
        See the HOW TO ADD NEW SLIDES block in main() for examples.
        """
        total = len(slides)
        print(f"\n{'═'*58}")
        print(f"  EXECUTING {total} SLIDE(S)")
        print(f"{'═'*58}\n")

        for i, slide in enumerate(slides, start=1):
            print(f"  ▶  Slide {i}/{total}: {slide.label}")
            self.move_to_slide(slide)

        print(f"\n{'═'*58}")
        print(f"  ALL {total} SLIDE(S) COMPLETE")
        print(f"{'═'*58}\n")

    # ── Cleanup ──────────────────────────────────────────────────────────────

    def disconnect(self):
        if self._physics_client is not None:
            p.disconnect(self._physics_client)
            self._physics_client = None
            print("[SIM] Disconnected from PyBullet.")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.disconnect()


# =============================================================================
# ── MOVEMENT SEQUENCE DEFINITION ──────────────────────────────────────────────
# =============================================================================

def build_sequence() -> List[Slide]:
    """
    Define every keyframe ("slide") the robot should visit.

    ── HOW TO ADD NEW SLIDES ──────────────────────────────────────────────────
    Append a new Slide(...) to the list below.

    Slide(
        joint_angles = [j1,  j2,    j3,   j4,    j5,    j6],
                         ^    ^      ^     ^      ^      ^
                         base shoulder elbow wrist_roll wrist_pitch wrist_roll2
                       (all values in RADIANS, limits ±1.5708 ≈ ±90°)

        gripper      = "open"  | "close" | "keep",
        duration     = <seconds to reach this pose>,
        label        = "your description here",
        pause_after  = <hold time in seconds after arriving>,
    )

    Tip: use  math.radians(deg)  to convert degrees → radians.
    Tip: gripper acts BEFORE the arm moves in that slide, so put
         "open" on the approach slide and "close" on the grasp slide.
    ── ─────────────────────────────────────────────────────────────────────────
    """
    RAD = math.radians  # shorthand

    return [
        # ── 0. Home / reset position ─────────────────────────────────────────
        Slide(
            joint_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            gripper      = "open",
            duration     = 2.0,
            label        = "Home – all zeros, gripper open",
            pause_after  = 0.5,
        ),

        # ── 1. Rotate base 45° left ───────────────────────────────────────────
        Slide(
            joint_angles = [RAD(45), 0.0, 0.0, 0.0, 0.0, 0.0],
            gripper      = "keep",
            duration     = 1.5,
            label        = "Base rotate 45° CCW",
            pause_after  = 0.3,
        ),

        # ── 2. Raise shoulder & elbow to reach forward ────────────────────────
        Slide(
            joint_angles = [RAD(45), RAD(-30), RAD(60), 0.0, RAD(20), 0.0],
            gripper      = "keep",
            duration     = 2.5,
            label        = "Shoulder up + elbow bend (reach position)",
            pause_after  = 0.4,
        ),

        # ── 3. Descend wrist toward pick target ──────────────────────────────
        Slide(
            joint_angles = [RAD(45), RAD(-45), RAD(75), 0.0, RAD(35), 0.0],
            gripper      = "open",
            duration     = 1.8,
            label        = "Pre-grasp descent – gripper open",
            pause_after  = 0.5,
        ),

        # ── 4. Close gripper to grasp object ────────────────────────────────
        Slide(
            joint_angles = [RAD(45), RAD(-45), RAD(75), 0.0, RAD(35), 0.0],
            gripper      = "close",
            duration     = 0.5,   # arm stays still while gripper closes
            label        = "GRASP – close gripper",
            pause_after  = 0.8,
        ),

        # ── 5. Lift object ───────────────────────────────────────────────────
        Slide(
            joint_angles = [RAD(45), RAD(-20), RAD(50), 0.0, RAD(15), 0.0],
            gripper      = "keep",
            duration     = 2.0,
            label        = "Lift object",
            pause_after  = 0.5,
        ),

        # ── 6. Swing to place position (opposite side) ───────────────────────
        Slide(
            joint_angles = [RAD(-60), RAD(-20), RAD(50), 0.0, RAD(15), 0.0],
            gripper      = "keep",
            duration     = 2.5,
            label        = "Swing to place position",
            pause_after  = 0.3,
        ),

        # ── 7. Lower to place ────────────────────────────────────────────────
        Slide(
            joint_angles = [RAD(-60), RAD(-40), RAD(70), 0.0, RAD(30), 0.0],
            gripper      = "keep",
            duration     = 1.8,
            label        = "Lower to place height",
            pause_after  = 0.5,
        ),

        # ── 8. Release object ────────────────────────────────────────────────
        Slide(
            joint_angles = [RAD(-60), RAD(-40), RAD(70), 0.0, RAD(30), 0.0],
            gripper      = "open",
            duration     = 0.5,
            label        = "RELEASE – open gripper",
            pause_after  = 0.8,
        ),

        # ── 9. Retract to home ───────────────────────────────────────────────
        Slide(
            joint_angles = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            gripper      = "keep",
            duration     = 2.5,
            label        = "Return to Home",
            pause_after  = 1.0,
        ),
    ]


# =============================================================================
# ── MAIN ENTRY POINT ──────────────────────────────────────────────────────────
# =============================================================================

def main():
    print("\n" + "═"*58)
    print("  ALVE-XXX Robot Arm  –  PyBullet Keyframe Sequencer")
    print("═"*58 + "\n")

    # ── 1. Validate all required files ───────────────────────────────────────
    try:
        validate_environment(PACKAGE_ROOT, URDF_SRC, MESHES_DIR)
    except FileNotFoundError as exc:
        sys.exit(str(exc))

    # ── 2. Patch the URDF (fix package:// mesh URIs) ─────────────────────────
    tmp_dir    = Path(tempfile.mkdtemp(prefix="alve_arm_"))
    patched    = patch_urdf(URDF_SRC, MESHES_DIR, tmp_dir)

    # ── 3. Build movement sequence ───────────────────────────────────────────
    sequence = build_sequence()
    print(f"[SEQ] Loaded {len(sequence)} slides.\n")

    # ── 4. Run simulation ────────────────────────────────────────────────────
    try:
        with AlveArmSimulation(gui=True) as sim:
            sim.connect()
            sim.load_robot(patched)
            sim.run_sequence(sequence)

            # Keep window open until the user closes it
            print("[SIM] Sequence finished.  Close the PyBullet window to exit.")
            if sim.gui:
                while True:
                    p.stepSimulation()
                    sim._apply_gripper_raw(sim._gripper_pos)
                    time.sleep(sim.SIM_TIMESTEP)

    except p.error as exc:
        print(f"\n[ERROR] PyBullet error: {exc}")
        raise
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        # Clean up the temporary patched URDF
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
