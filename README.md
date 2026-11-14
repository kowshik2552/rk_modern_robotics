# rk_modern_robotics
> **Documentation:** [View / Download the Full Reference Manual (PDF)](./rk_modern_robotics_manual.pdf)
A Python library for screw-theory robotics kinematics and dynamics based on Kevin M. Lynch and Frank C. Park's ***Modern Robotics: Mechanics, Planning, and Control***, built with **SymPy** and **NumPy** to support both **closed-form symbolic derivation** and **numeric evaluation**.

## Key Features

* **Dual Symbolic & Numeric Execution:** Pass SymPy symbols to derive closed-form transformation matrices, Jacobians, mass matrices $M(\theta)$, Coriolis/centripetal vectors $c(\theta, \dot{\theta})$, and gravity vectors $g(\theta)$—or pass standard Python lists / NumPy arrays for automatic floating-point evaluation.
* **Familiar *Modern Robotics* API:** Mirrors the function names, argument orders, and conventions of the official `modern_robotics` package.
* **Analytical & Numerical Projections:** Supports symbolic Gram-Schmidt orthonormalization onto $SO(3)$ and $SE(3)$ alongside SVD-based projection for numeric matrices.
* **Recursive Newton-Euler Dynamics:** Full symbolic and numeric support for inverse dynamics, mass matrix computation, velocity-product forces, gravity torques, end-effector wrench mapping, and forward dynamics simulations.

---

## Installation

```bash
pip install rk_modern_robotics
```

### Dependencies
* `sympy`
* `numpy`

---

## Quick Start

> **Important:** Whenever you perform symbolic computations, **always import `sympy` and explicitly declare your symbolic variables using `sp.symbols(...)`** before passing them into `rk_modern_robotics` functions.

### 1. Symbolic Forward Kinematics & Space Jacobian (2R Planar Arm)

```python
import sympy as sp
import rk_modern_robotics as rk

# Always declare symbolic variables explicitly using SymPy
theta1, theta2, L1, L2 = sp.symbols('theta1 theta2 L1 L2')

# Home configuration M and Space screw axes Slist
M = [[1, 0, 0, L1 + L2],
     [0, 1, 0,       0],
     [0, 0, 1,       0],
     [0, 0, 0,       1]]

Slist = [[0,   0],
         [0,   0],
         [1,   1],
         [0,   0],
         [0, -L1],
         [0,   0]]

T = rk.FKinSpace(M, Slist, [theta1, theta2])
Js = rk.JacobianSpace(Slist, [theta1, theta2])

sp.pprint(T)
sp.pprint(Js)
```

### 2. Symbolic Dynamics (Deriving $M(\theta)$ and $g(\theta)$)

```python
import sympy as sp
import rk_modern_robotics as rk

# Always declare symbolic variables explicitly using SymPy
theta, m, L, g = sp.symbols('theta m L g')

Mlist = [
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -L], [0, 0, 0, 1]],
    sp.eye(4)
]
Glist = [sp.diag(0, 0, 0, m, m, m)]
Slist = [[0], [1], [0], [0], [0], [0]]

M_theta = rk.MassMatrix([theta], Mlist, Glist, Slist)
g_theta = rk.GravityForces([theta], [0, 0, -g], Mlist, Glist, Slist)

print("Mass Matrix M(theta):\n", M_theta)
print("\nGravity Vector g(theta):\n", g_theta)
```

### 3. Numeric Inverse Kinematics

*Note: `IKinBody` and `IKinSpace` are iterative Newton-Raphson solvers and require purely numeric inputs (no free SymPy symbols).*

```python
import rk_modern_robotics as rk

Blist = [[ 0, 0,   0],
         [ 0, 0,   0],
         [-1, 0,   1],
         [ 2, 0,   0],
         [ 0, 1,   0],
         [ 0, 0, 0.1]]

M = [[-1, 0,  0, 0],
     [ 0, 1,  0, 6],
     [ 0, 0, -1, 2],
     [ 0, 0,  0, 1]]

T = [[0, 1,  0,     -5],
     [1, 0,  0,      4],
     [0, 0, -1, 1.6858],
     [0, 0,  0,      1]]

thetalist, success = rk.IKinBody(Blist, M, T, [1.5, 2.5, 3], eomg=0.01, ev=0.001)
print("Joint Angles:\n", thetalist)
print("Converged:", success)
```

---

## Implemented Functions

### Basic Utilities

| Function | Description | Mode |
| :--- | :--- | :--- |
| `NearZero(z)` | Checks if scalar $|z| < 10^{-6}$ (or symbolically equal to `0`) | Symbolic & Numeric |

### Rigid-Body Motions ($SO(3)$, $SE(3)$, $\mathfrak{so}(3)$, $\mathfrak{se}(3)$)

| Function | Description | Mode |
| :--- | :--- | :--- |
| `RotInv(R)` | Transpose/inverse of a rotation matrix $R \in SO(3)$ | Symbolic & Numeric |
| `VecToso3(omega)` | Converts a 3-vector $\omega$ into a $3 \times 3$ skew-symmetric matrix $[\omega]$ | Symbolic & Numeric |
| `so3ToVec(so3mat)` | Extracts a 3-vector $\omega$ from a skew-symmetric matrix $[\omega]$ | Symbolic & Numeric |
| `AxisAng3(expc3)` | Extracts rotation axis $\hat{\omega}$ and angle $\theta$ from $\hat{\omega}\theta$ | Symbolic & Numeric |
| `MatrixExp3(so3mat)` | Matrix exponential $e^{[\hat{\omega}]\theta} \in SO(3)$ (Rodrigues' formula) | Symbolic & Numeric |
| `MatrixLog3(R)` | Matrix logarithm $[\hat{\omega}]\theta \in \mathfrak{so}(3)$ of a rotation matrix $R$ | Symbolic & Numeric |
| `RpToTrans(R, p)` | Builds a $4 \times 4$ homogeneous transformation matrix $T \in SE(3)$ from $(R, p)$ | Symbolic & Numeric |
| `TransToRp(T)` | Extracts rotation matrix $R$ and position vector $p$ from $T \in SE(3)$ | Symbolic & Numeric |
| `TransInv(T)` | Analytical inverse $T^{-1}$ of a homogeneous transformation matrix | Symbolic & Numeric |
| `VecTose3(V)` | Converts a 6D spatial velocity/twist $V$ into a $4 \times 4$ matrix $[V] \in \mathfrak{se}(3)$ | Symbolic & Numeric |
| `se3ToVec(se3mat)` | Extracts a 6D twist vector $V$ from a $4 \times 4$ matrix $[V] \in \mathfrak{se}(3)$ | Symbolic & Numeric |
| `Adjoint(T)` | Computes the $6 \times 6$ adjoint representation $[\text{Ad}_T]$ of $T \in SE(3)$ | Symbolic & Numeric |
| `ScrewToAxis(q, s, h)` | Constructs a normalized screw axis $\mathcal{S}$ from point $q$, unit axis $\hat{s}$, and pitch $h$ | Symbolic & Numeric |
| `AxisAng6(expc6)` | Extracts normalized screw axis $\mathcal{S}$ and angle $\theta$ from $\mathcal{S}\theta$ | Symbolic & Numeric |
| `MatrixExp6(se3mat)` | Matrix exponential $e^{[\mathcal{S}]\theta} \in SE(3)$ | Symbolic & Numeric |
| `MatrixLog6(T)` | Matrix logarithm $[\mathcal{S}]\theta \in \mathfrak{se}(3)$ of a transformation matrix $T$ | Symbolic & Numeric |
| `ProjectToSO3(mat)` | Projects a $3 \times 3$ matrix onto $SO(3)$ (SVD for numeric, Gram-Schmidt for symbolic) | Symbolic & Numeric |
| `ProjectToSE3(mat)` | Projects a $4 \times 4$ matrix onto $SE(3)$ | Symbolic & Numeric |
| `DistanceToSO3(mat)` | Frobenius distance from a $3 \times 3$ matrix to $SO(3)$ | Symbolic & Numeric |
| `DistanceToSE3(mat)` | Distance from a $4 \times 4$ matrix to $SE(3)$ | Symbolic & Numeric |
| `TestIfSO3(mat)` | Returns `True` if matrix is within $10^{-3}$ of $SO(3)$ | Numeric |
| `TestIfSE3(mat)` | Returns `True` if matrix is within $10^{-3}$ of $SE(3)$ | Numeric |

### Kinematics

| Function | Description | Mode |
| :--- | :--- | :--- |
| `FKinBody(M, Blist, thetalist)` | Forward kinematics in the end-effector (body) frame | Symbolic & Numeric |
| `FKinSpace(M, Slist, thetalist)` | Forward kinematics in the fixed (space) frame | Symbolic & Numeric |
| `JacobianBody(Blist, thetalist)` | Computes the $6 \times n$ body Jacobian $J_b(\theta)$ | Symbolic & Numeric |
| `JacobianSpace(Slist, thetalist)` | Computes the $6 \times n$ space Jacobian $J_s(\theta)$ | Symbolic & Numeric |
| `IKinBody(Blist, M, T, thetalist0, eomg, ev)` | Iterative Newton-Raphson inverse kinematics in the body frame | Numeric Only |
| `IKinSpace(Slist, M, T, thetalist0, eomg, ev)` | Iterative Newton-Raphson inverse kinematics in the space frame | Numeric Only |

### Open-Chain Dynamics

| Function | Description | Mode |
| :--- | :--- | :--- |
| `ad(V)` | Computes the $6 \times 6$ spatial Lie bracket matrix $[\text{ad}_V]$ | Symbolic & Numeric |
| `InverseDynamics(...)` | Computes joint torques $\tau$ via Recursive Newton-Euler inverse dynamics | Symbolic & Numeric |
| `MassMatrix(thetalist, Mlist, Glist, Slist)` | Computes the $n \times n$ symmetric positive-definite mass matrix $M(\theta)$ | Symbolic & Numeric |
| `VelQuadraticForces(thetalist, dthetalist, Mlist, Glist, Slist)` | Computes Coriolis and centripetal torque vector $c(\theta, \dot{\theta})$ | Symbolic & Numeric |
| `GravityForces(thetalist, g, Mlist, Glist, Slist)` | Computes joint torques $g(\theta)$ required to resist gravity | Symbolic & Numeric |
| `EndEffectorForces(thetalist, Ftip, Mlist, Glist, Slist)` | Computes joint torques $J^T(\theta)\mathcal{F}_{tip}$ required to support an end-effector wrench | Symbolic & Numeric |
| `ForwardDynamics(...)` | Solves for joint accelerations $\ddot{\theta}$ given applied torques $\tau$ | Symbolic & Numeric |
| `EulerStep(thetalist, dthetalist, ddthetalist, dt)` | First-order Euler integration step for joint positions and velocities | Symbolic & Numeric |
| `InverseDynamicsTrajectory(...)` | Computes required joint torque trajectory along a reference trajectory | Symbolic & Numeric |
| `ForwardDynamicsTrajectory(...)` | Simulates robot motion over time given an open-loop torque trajectory | Symbolic & Numeric |

---

## Attribution & License

This library is adapted from the official [ModernRobotics](https://github.com/NxRLab/ModernRobotics) Python codebase by **NxRLab** (accompanying *Modern Robotics: Mechanics, Planning, and Control* by Kevin M. Lynch and Frank C. Park, Cambridge University Press, 2017).

Released under the **MIT License** — Copyright (c) 2026 Ravi Kowshik. See the `LICENSE` file for full details.