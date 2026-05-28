"""
3D Molecular Dynamics Simulation of an Ideal Gas
Year-End R Lab Project

Simulates N identical point-particles in a cubic box using perfectly elastic
wall collisions (no inter-particle interactions, no gravity).
Demonstrates the Ideal Gas Law: P = kB * (N*T / V)

Parts:
  I   – Velocity initialisation (Maxwell-Boltzmann, momentum zeroing, rescaling)
  II  – Simulation engine (kinematics + wall collisions)
  III – Pressure from impulse accumulation
  IV  – Temperature sweep → P vs NT/V plot + 3D animation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)

# ──────────────────────────────────────────────────────────────────────────────
# Physical constants  (SI units)
# ──────────────────────────────────────────────────────────────────────────────
kB = 1.380649e-23   # J/K  – Boltzmann constant
m  = 6.63e-26       # kg   – mass of one argon atom (~40 u)

# ──────────────────────────────────────────────────────────────────────────────
# System parameters
# ──────────────────────────────────────────────────────────────────────────────
L  = 1e-8           # m   – cubic box side length (10 nm)
N  = 100            # number of particles
V  = L**3           # m³  – volume

dt      = 1e-13     # s   – time step  (~1/400 of box-crossing time at 300 K)
n_eq    = 500       # equilibration steps (discarded)
n_steps = 3000      # production steps   (pressure measurement window)

# ══════════════════════════════════════════════════════════════════════════════
# PART I  –  Velocity initialisation
# ══════════════════════════════════════════════════════════════════════════════

def init_positions(N, L):
    """Uniformly random positions inside the box [0, L]^3."""
    return np.random.uniform(0.0, L, (N, 3))


def calc_temperature(v):
    """
    Kinetic theory (3D):  (3/2) kB T = (1/2) m <v^2>
    => T = m <v^2> / (3 kB)
    """
    v_sq_mean = np.mean(np.sum(v**2, axis=1))
    return m * v_sq_mean / (3.0 * kB)


def init_velocities(N, T):
    """
    Draw velocities from the Maxwell-Boltzmann distribution at temperature T.

    Step 1 – Random sampling: normal(0, sigma) per component, sigma = sqrt(kB T / m)
    Step 2 – Zero net momentum so the gas does not drift
    Step 3 – Rescale to match the target temperature exactly
    """
    # Step 1 – Random sampling
    sigma = np.sqrt(kB * T / m)
    v = np.random.normal(0.0, sigma, (N, 3))

    # Step 2 – Zero net momentum  v_i -= mean(v)
    v -= v.mean(axis=0)

    # Step 3 – Temperature rescaling  v *= sqrt(T_target / T_current)
    lam = np.sqrt(T / calc_temperature(v))
    v  *= lam
    return v


# ══════════════════════════════════════════════════════════════════════════════
# PART II  –  Microscopic simulation engine
# ══════════════════════════════════════════════════════════════════════════════

def step(pos, vel):
    """
    Advance all particles by one time step dt.

    Kinematics:  r_new = r_old + v * dt
    Boundaries:  elastic bounce off each of the 6 walls.
                 Impulse on wall = |2 m v_perp| per collision.

    Returns
    -------
    pos      : updated positions  (N, 3)
    vel      : updated velocities (N, 3)
    dp_total : total momentum transferred to walls this step (scalar)
    """
    pos = pos + vel * dt           # kinematics (vectorised over all N particles)
    dp_total = 0.0

    for axis in range(3):
        # ── low wall  (coordinate < 0) ─────────────────────────────────────
        lo = pos[:, axis] < 0.0
        if lo.any():
            dp_total        += np.sum(2.0 * m * np.abs(vel[lo, axis]))
            vel[lo, axis]   *= -1.0
            pos[lo, axis]    = -pos[lo, axis]           # reflect back into box

        # ── high wall  (coordinate > L) ────────────────────────────────────
        hi = pos[:, axis] > L
        if hi.any():
            dp_total        += np.sum(2.0 * m * np.abs(vel[hi, axis]))
            vel[hi, axis]   *= -1.0
            pos[hi, axis]    = 2.0 * L - pos[hi, axis] # reflect back into box

    return pos, vel, dp_total


# ══════════════════════════════════════════════════════════════════════════════
# PART III  –  Macroscopic pressure from impulse accumulation
# ══════════════════════════════════════════════════════════════════════════════

def measure_pressure(pos, vel):
    """
    Run the production phase and compute pressure via impulse method.

      P = sum(Delta_p) / (A * Delta_t)   with A = 6 L^2  (all 6 faces)

    Returns (P_Pa, T_final_K).
    """
    total_dp = 0.0
    for _ in range(n_steps):
        pos, vel, dp = step(pos, vel)
        total_dp += dp

    A       = 6.0 * L**2        # total surface area of cube
    Delta_t = n_steps * dt
    P       = total_dp / (A * Delta_t)
    return P, calc_temperature(vel)


def run_at_temperature(T):
    """Initialise, equilibrate, then measure pressure at temperature T."""
    pos = init_positions(N, L)
    vel = init_velocities(N, T)

    # Equilibration – let the system settle
    for _ in range(n_eq):
        pos, vel, _ = step(pos, vel)

    return measure_pressure(pos, vel)


# ══════════════════════════════════════════════════════════════════════════════
# PART IV  –  Temperature sweep → derive ideal gas law
# ══════════════════════════════════════════════════════════════════════════════

T_targets = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], dtype=float)

print("=" * 60)
print("  3D Ideal Gas MD – pressure sweep")
print("=" * 60)
P_vals = []
T_vals = []

for T in T_targets:
    P, T_m = run_at_temperature(T)
    P_vals.append(P)
    T_vals.append(T_m)
    print(f"  T_target = {int(T):4d} K | T_measured = {T_m:6.1f} K | P = {P:.4e} Pa")

P_vals = np.array(P_vals)
T_vals = np.array(T_vals)

# ── Linear fit:  P = k_sim * (N T / V) ────────────────────────────────────────
x = N * T_vals / V       # units: K m^{-3}

slope, intercept = np.polyfit(x, P_vals, 1)

print()
print(f"  Simulation Gas Constant (slope) = {slope:.4e} J/K")
print(f"  Boltzmann constant kB           = {kB:.4e} J/K")
print(f"  Relative error                  = {abs(slope - kB) / kB * 100:.2f} %")
print("=" * 60)

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1  –  P vs NT/V  (ideal gas law verification)
# ══════════════════════════════════════════════════════════════════════════════

fig1, ax = plt.subplots(figsize=(8, 5))

ax.scatter(x, P_vals, color='royalblue', s=70, zorder=5, label='Simulation data')

x_fit = np.linspace(0, x.max() * 1.05, 300)
ax.plot(x_fit, slope * x_fit + intercept, 'r--', linewidth=2,
        label=f'Linear fit   $k_\\mathrm{{sim}}$ = {slope:.3e} J/K')
ax.plot(x_fit, kB * x_fit, color='gray', linestyle=':', linewidth=1.5,
        label=f'Ideal gas    $k_B$ = {kB:.3e} J/K')

ax.set_xlabel(r'$NT\,/\,V$   (K m$^{-3}$)', fontsize=13)
ax.set_ylabel('Pressure  (Pa)', fontsize=13)
ax.set_title(r'Ideal Gas Law:   $P = k_B\,\dfrac{N\,T}{V}$', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig('ideal_gas_law.png', dpi=150)
print("\nSaved  ideal_gas_law.png")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 2  –  3D animation of gas particles
# ══════════════════════════════════════════════════════════════════════════════

print("Generating 3D animation …")

T_anim = 300                # K – temperature shown in animation
pos_a  = init_positions(N, L)
vel_a  = init_velocities(N, T_anim)

n_frames        = 60
steps_per_frame = 8

# Pre-compute frames
frames = []
for _ in range(n_frames):
    for _ in range(steps_per_frame):
        pos_a, vel_a, _ = step(pos_a, vel_a)
    frames.append(pos_a.copy())

# ── Build figure ───────────────────────────────────────────────────────────────
fig2 = plt.figure(figsize=(7, 7))
ax3  = fig2.add_subplot(111, projection='3d')

# Draw box wireframe
corners = [(s, e) for s, e in [
    ((0,0,0),(L,0,0)), ((0,0,0),(0,L,0)), ((0,0,0),(0,0,L)),
    ((L,L,L),(0,L,L)), ((L,L,L),(L,0,L)), ((L,L,L),(L,L,0)),
    ((L,0,0),(L,L,0)), ((L,0,0),(L,0,L)),
    ((0,L,0),(L,L,0)), ((0,L,0),(0,L,L)),
    ((0,0,L),(L,0,L)), ((0,0,L),(0,L,L)),
]]
for s, e in corners:
    ax3.plot3D(*zip(s, e), 'k-', alpha=0.25, linewidth=0.8)

sc = ax3.scatter([], [], [], s=20, c='steelblue', alpha=0.80, depthshade=True)

ax3.set_xlim(0, L); ax3.set_ylim(0, L); ax3.set_zlim(0, L)
ax3.set_xlabel('x (m)'); ax3.set_ylabel('y (m)'); ax3.set_zlabel('z (m)')
ax3.set_title(f'3D Ideal Gas MD   (N = {N},  T = {T_anim} K)', pad=10)

time_text = ax3.text2D(0.02, 0.95, '', transform=ax3.transAxes, fontsize=9)

def update(frame_idx):
    p = frames[frame_idx]
    sc._offsets3d = (p[:, 0], p[:, 1], p[:, 2])
    time_text.set_text(f'frame {frame_idx + 1}/{n_frames}')
    return sc, time_text

ani = FuncAnimation(fig2, update, frames=n_frames, interval=80, blit=False)
ani.save('gas_animation.gif', writer='pillow', fps=12)
print("Saved  gas_animation.gif")

plt.show()
print("\nDone!")
