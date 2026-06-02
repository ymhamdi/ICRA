# 🦾 ALVE-XXX • Intelligent Robotic Vision & Kinematics Engine
> **High-Fidelity 6-Axis Digital Twin with Real-Time Computer Vision & Closed-Loop Pick-and-Place Automation**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyBullet-Physics-FF6F00?style=for-the-badge" alt="PyBullet" />
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV" />
  <img src="https://img.shields.io/badge/Repository-Private-red?style=for-the-badge" alt="Private Repo" />
</p>

---

## 🏗️ System Architecture Matrix
┌──────────────────────┐      ┌──────────────────────┐      ┌──────────────────────┐
│   SYNTHETIC CAMERA   │      │   OPENCV PROCESSOR   │      │  INVERSE KINEMATICS  │
│  (RGB-D Image Feed)  │ ───> │ (HSV Color Isolation)│ ───> │ (3D World Space Map) │
└──────────────────────┘      └──────────────────────┘      └──────────────────────┘
│
▼
┌──────────────────────┐      ┌──────────────────────┐      ┌──────────────────────┐
│     PICK ENGINE      │      │    TARGET OBJECT     │      │   6-AXIS ACTUATION   │
│  (Parallel Gripper)  │ <─── │ (Grasp & Transport)  │ <─── │  (High-Torque Loop)  │
└──────────────────────┘      └──────────────────────┘      └──────────────────────┘

## 🛠️ Detailed Component Architecture

### 1. Vision Subsystem (OpenCV)
The system mounts a virtual camera within the environment coordinates. The captured stream is processed through an isolated pipeline:
* **Color Isolation:** Converts raw RGB data arrays to the **HSV (Hue, Saturation, Value)** color space to ignore shifting simulation lighting.
* **Binarization:** Applies spatial threshold masking (`cv2.inRange`) to filter everything except the target block's color signatures.
* **Centroid Tracking:** Calculates moments on the largest detected contour boundary to extract pixel coordinates $(cx, cy)$.

### 2. Kinematics & Mapping Engine (PyBullet)
Once pixel positions are secured, the software handles world translations:
* **Depth Decoupling:** Reconstructs the target's absolute 3D position vector $(X_w, Y_w, Z_w)$ by intersecting the standard projection matrices with the camera's depth buffer.
* **IK Solver:** Feeds target coordinate matrices into PyBullet’s Inverse Kinematics module (`p.calculateInverseKinematics`), solving joint variables smoothly.

### 3. End-Effector Command Sequence ("Grop")
Because physics engines natively struggle with passive mimic constraints, the controller bypasses typical configuration limitations by manually binding the parallel gripper links. Closing the master driver joint (`r_grip_joint`) mirrors velocity and position vectors directly onto slave sub-links, maintaining solid grasp torque on physical entities.

---

## 📂 Repository Topology

The framework maps exactly to the file structure configuration outlined below:

```text
ICRA/
┃
┣ 📄 alve_arm_simulation.py      # Main autonomous vision & control application
┣ 📄 README.md                   # System documentation and deployment blueprints
┃
┗ 📂 alve-xxx-robot-arm-urdf-main/
  ┣ 📂 config/                   # Joint mappings and configuration manifests
  ┣ 📂 launch/                   # Deployment automation routines
  ┣ 📂 meshes/                   # Solid model geometries (.STL structural components)
  ┗ 📂 urdf/                     # Kinematic structural trees (ALVE-XXX URDF Core)
