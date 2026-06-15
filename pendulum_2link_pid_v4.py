from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pybullet as p

import matplotlib

try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt


@dataclass
class PIDGains:
    kp: np.ndarray
    ki: np.ndarray
    kd: np.ndarray


@dataclass
class SimConfig:
    dt: float = 1.0 / 240.0
    tmax: float = 15.0
    gui: bool = False
    q0: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0], dtype=float))
    q_des: np.ndarray = field(default_factory=lambda: np.array([0.8, -0.6], dtype=float))
    torque_limits: np.ndarray = field(default_factory=lambda: np.array([140.0, 110.0], dtype=float))
    gains: PIDGains = field(
        default_factory=lambda: PIDGains(
            kp=np.array([110.0, 80.0], dtype=float),
            ki=np.array([18.0, 16.0], dtype=float),
            kd=np.array([21.0, 9.0], dtype=float),
        )
    )


def wrap_to_pi(angle: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def joint_error(q_des: np.ndarray, q: np.ndarray) -> np.ndarray:
    return wrap_to_pi(q_des - q)


def clamp(x: np.ndarray, limits: np.ndarray) -> np.ndarray:
    return np.clip(x, -limits, limits)


def build_sim(config: SimConfig, urdf_path: Path) -> tuple[int, int, tuple[int, int]]:
    if not urdf_path.exists():
        raise FileNotFoundError(
            f"URDF file not found: {urdf_path}. Put two_link_pendulum.urdf next to this script."
        )

    client = p.connect(p.GUI if config.gui else p.DIRECT)
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(config.dt)
    p.setRealTimeSimulation(0)
    p.setPhysicsEngineParameter(numSolverIterations=220)

    robot = p.loadURDF(str(urdf_path), useFixedBase=True)

    joint_1 = 1
    joint_2 = 2

    for j in (joint_1, joint_2):
        p.setJointMotorControl2(robot, j, controlMode=p.VELOCITY_CONTROL, force=0)
        p.changeDynamics(robot, j, linearDamping=0.0, angularDamping=0.0)

    return client, robot, (joint_1, joint_2)


def run_simulation(config: SimConfig, urdf_path: Path) -> dict[str, np.ndarray]:
    client, robot, (joint_1, joint_2) = build_sim(config, urdf_path)

    q0 = np.asarray(config.q0, dtype=float).copy()
    q_des = np.asarray(config.q_des, dtype=float).copy()
    gains = config.gains
    torque_limits = np.asarray(config.torque_limits, dtype=float).copy()

    p.resetJointState(robot, joint_1, float(q0[0]), 0.0)
    p.resetJointState(robot, joint_2, float(q0[1]), 0.0)

    n_steps = int(config.tmax / config.dt)
    t_log = np.zeros(n_steps + 1)
    q_log = np.zeros((n_steps + 1, 2))
    dq_log = np.zeros((n_steps + 1, 2))
    tau_log = np.zeros((n_steps + 1, 2))
    ee_log = np.zeros((n_steps + 1, 3))

    e_int = np.zeros(2)
    e_int_limit = np.array([2.5, 2.5], dtype=float)
    e_prev = joint_error(q_des, q0)

    def log_state(i: int, t: float, tau: Sequence[float] | None = None) -> None:
        s1 = p.getJointState(robot, joint_1)
        s2 = p.getJointState(robot, joint_2)
        q = np.array([s1[0], s2[0]], dtype=float)
        dq = np.array([s1[1], s2[1]], dtype=float)
        ls = p.getLinkState(robot, 3, computeForwardKinematics=True)
        ee = np.array(ls[4], dtype=float)
        t_log[i] = t
        q_log[i] = q
        dq_log[i] = dq
        ee_log[i] = ee
        if tau is not None:
            tau_log[i] = np.asarray(tau, dtype=float)

    # First control action before the first integration step
    q = q0.copy()
    dq = np.zeros(2)
    e = joint_error(q_des, q)
    tau0 = gains.kp * e + gains.ki * e_int - gains.kd * dq
    tau0 = clamp(tau0, torque_limits)
    p.setJointMotorControl2(robot, joint_1, controlMode=p.TORQUE_CONTROL, force=float(tau0[0]))
    p.setJointMotorControl2(robot, joint_2, controlMode=p.TORQUE_CONTROL, force=float(tau0[1]))
    log_state(0, 0.0, tau=tau0)

    for i in range(1, n_steps + 1):
        s1 = p.getJointState(robot, joint_1)
        s2 = p.getJointState(robot, joint_2)
        q = np.array([s1[0], s2[0]], dtype=float)
        dq = np.array([s1[1], s2[1]], dtype=float)

        e = joint_error(q_des, q)
        de = (e - e_prev) / config.dt
        e_prev = e

        e_int += e * config.dt
        e_int = np.clip(e_int, -e_int_limit, e_int_limit)

        tau_unsat = gains.kp * e + gains.ki * e_int - gains.kd * dq
        tau = clamp(tau_unsat, torque_limits)

        # Lightweight anti-windup for faster settling.
        saturated = np.abs(tau_unsat - tau) > 1e-9
        if np.any(saturated):
            e_int[saturated] *= 0.96

        p.setJointMotorControl2(robot, joint_1, controlMode=p.TORQUE_CONTROL, force=float(tau[0]))
        p.setJointMotorControl2(robot, joint_2, controlMode=p.TORQUE_CONTROL, force=float(tau[1]))

        p.stepSimulation()

        if config.gui:
            time.sleep(config.dt)

        log_state(i, i * config.dt, tau=tau)

    p.disconnect(client)

    return {"t": t_log, "q": q_log, "dq": dq_log, "tau": tau_log, "ee": ee_log}


def plot_results(result: dict[str, np.ndarray], config: SimConfig, out_path: Path) -> None:
    t = result["t"]
    q = result["q"]
    tau = result["tau"]
    ee = result["ee"]

    fig = plt.figure(figsize=(11, 10))

    ax1 = fig.add_subplot(4, 1, 1)
    ax1.plot(t, q[:, 0], label=r"$q_1$")
    ax1.plot(t, q[:, 1], label=r"$q_2$")
    ax1.plot([t[0], t[-1]], [config.q_des[0], config.q_des[0]], "--", label=r"$q_{1,des}$")
    ax1.plot([t[0], t[-1]], [config.q_des[1], config.q_des[1]], "--", label=r"$q_{2,des}$")
    ax1.set_ylabel("Angle [rad]")
    ax1.grid(True)
    ax1.legend(loc="best")

    ax2 = fig.add_subplot(4, 1, 2)
    ax2.plot(t, tau[:, 0], label=r"$\tau_1$")
    ax2.plot(t, tau[:, 1], label=r"$\tau_2$")
    ax2.set_ylabel("Torque [N·m]")
    ax2.grid(True)
    ax2.legend(loc="best")

    ax3 = fig.add_subplot(4, 1, 3)
    ax3.plot(ee[:, 0], ee[:, 2], label="EEF path")
    ax3.set_xlabel("x [m]")
    ax3.set_ylabel("z [m]")
    ax3.grid(True)
    ax3.legend(loc="best")
    ax3.set_aspect("equal", adjustable="box")

    ax4 = fig.add_subplot(4, 1, 4)
    err = np.abs(joint_error(config.q_des, q))
    ax4.plot(t, err[:, 0], label=r"$|e_1|$")
    ax4.plot(t, err[:, 1], label=r"$|e_2|$")
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Abs. error [rad]")
    ax4.grid(True)
    ax4.legend(loc="best")

    fig.suptitle("Two-link pendulum: PID torque control")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-link pendulum PID in PyBullet")
    parser.add_argument("--gui", action="store_true", help="Run with PyBullet GUI")
    parser.add_argument("--tmax", type=float, default=15.0, help="Simulation time [s]")
    parser.add_argument(
        "--qdes",
        type=float,
        nargs=2,
        default=[0.8, -0.6],
        metavar=("Q1", "Q2"),
        help="Desired joint angles [rad]",
    )
    parser.add_argument(
        "--q0",
        type=float,
        nargs=2,
        default=[0.0, 0.0],
        metavar=("Q1", "Q2"),
        help="Initial joint angles [rad]",
    )
    parser.add_argument(
        "--urdf",
        type=str,
        default="two_link_pendulum.urdf",
        help="Path to URDF file",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default="two_link_pid_result.png",
        help="Output plot file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SimConfig(
        gui=bool(args.gui),
        tmax=float(args.tmax),
        q0=np.array(args.q0, dtype=float),
        q_des=np.array(args.qdes, dtype=float),
    )

    urdf_path = Path(args.urdf).resolve()
    result = run_simulation(config, urdf_path)
    out_path = Path(args.outfile).resolve()
    plot_results(result, config, out_path)

    q_final = result["q"][-1]
    err = joint_error(config.q_des, q_final)
    err_deg = np.degrees(err)

    print("Simulation finished.")
    print(f"Used URDF: {urdf_path}")
    print(f"Saved plot: {out_path}")
    print(f"Final joint angles: q1={q_final[0]:.4f} rad, q2={q_final[1]:.4f} rad")
    print(f"Target joint angles: q1={config.q_des[0]:.4f} rad, q2={config.q_des[1]:.4f} rad")
    print(f"Final error: {err} rad")
    print(f"Final error: {err_deg} deg")


if __name__ == "__main__":
    main()
