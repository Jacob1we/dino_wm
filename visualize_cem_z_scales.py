"""
Vergleich: sigma_scale_z = 1, 2, 3 — Seitenansicht (XZ) + Z-Marginal.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
from scipy.stats import norm
import os

# --- Dataset-Statistiken (8D) ---
ACTION_MEAN = np.array([0.400, 0.025, 0.050, 0.50, 0.400, 0.025, 0.050, 0.50])
ACTION_STD  = np.array([0.121, 0.158, 0.070, 0.50, 0.121, 0.158, 0.070, 0.50])
VAR_SCALE = 1.0
IX, IY, IZ = 4, 5, 6  # End-EEF indices in 8D

# Workspace Bounds
WS_LOWER = np.array([0.05, -0.40, 0.00])
WS_UPPER = np.array([0.80,  0.45, 0.12])

N_SAMPLES = 3000
np.random.seed(42)

Z_SCALES = [1.0, 2.0, 3.0]
COLORS = ['tab:orange', 'tab:green', 'tab:blue']

# ============================================================================
# Samples für alle drei z-Scales generieren
# ============================================================================
all_samples_raw = {}
all_samples_clamp = {}

for zs in Z_SCALES:
    sigma_norm = VAR_SCALE * np.array([1, 1, zs, 1, 1, 1, zs, 1])
    samples = np.random.randn(N_SAMPLES, 8) * sigma_norm
    samples_world = samples * ACTION_STD + ACTION_MEAN
    clamped = samples_world.copy()
    clamped[:, IX] = np.clip(clamped[:, IX], WS_LOWER[0], WS_UPPER[0])
    clamped[:, IY] = np.clip(clamped[:, IY], WS_LOWER[1], WS_UPPER[1])
    clamped[:, IZ] = np.clip(clamped[:, IZ], WS_LOWER[2], WS_UPPER[2])
    all_samples_raw[zs] = samples_world
    all_samples_clamp[zs] = clamped

mu_x = ACTION_MEAN[IX]
mu_z = ACTION_MEAN[IZ]

# ============================================================================
# PLOT: 3 Seitenansichten (XZ) nebeneinander
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(21, 7))
fig.suptitle("CEM Sampling — Seitenansicht (XZ) bei sigma_scale_z = 1, 2, 3",
             fontsize=16, fontweight='bold')

for i, zs in enumerate(Z_SCALES):
    ax = axes[i]
    samples_raw = all_samples_raw[zs]
    samples_clamp = all_samples_clamp[zs]
    color = COLORS[i]

    phys_sigma_x = VAR_SCALE * 1.0 * ACTION_STD[IX]
    phys_sigma_z = VAR_SCALE * zs  * ACTION_STD[IZ]

    # Workspace Bounds
    ws_rect = patches.Rectangle(
        (WS_LOWER[0], WS_LOWER[2]),
        WS_UPPER[0] - WS_LOWER[0],
        WS_UPPER[2] - WS_LOWER[2],
        linewidth=2.5, edgecolor='red', facecolor='red', alpha=0.08,
        label=f'Action Bounds z∈[{WS_LOWER[2]:.2f}, {WS_UPPER[2]:.2f}]'
    )
    ax.add_patch(ws_rect)

    # Tisch + Cube Zone
    ax.axhline(y=0.04, color='saddlebrown', linewidth=3, alpha=0.6,
               label='Tisch (~0.04 m)')
    ax.axhspan(0, 0.04, color='saddlebrown', alpha=0.1)
    ax.axhspan(0.04, 0.09, color='green', alpha=0.08, label='Cube-Zone')

    # Samples
    ax.scatter(samples_raw[:, IX], samples_raw[:, IZ],
               s=3, alpha=0.12, c='gray', label='vor Clamping')
    ax.scatter(samples_clamp[:, IX], samples_clamp[:, IZ],
               s=5, alpha=0.3, c=color, label='nach Clamping')

    # Konfidenz-Ellipsen
    for n_std, alpha in [(1, 0.5), (2, 0.3), (3, 0.15)]:
        e = Ellipse((mu_x, mu_z), 2*n_std*phys_sigma_x, 2*n_std*phys_sigma_z,
                    facecolor='none', edgecolor='darkblue', linewidth=1.5,
                    linestyle='--', alpha=alpha,
                    label=f'{n_std}σ' if alpha == 0.5 else None)
        ax.add_patch(e)

    ax.plot(mu_x, mu_z, 'k+', markersize=15, markeredgewidth=2)

    # In-bounds Statistik
    in_x = (samples_raw[:, IX] >= WS_LOWER[0]) & (samples_raw[:, IX] <= WS_UPPER[0])
    in_z = (samples_raw[:, IZ] >= WS_LOWER[2]) & (samples_raw[:, IZ] <= WS_UPPER[2])
    pct_xz = 100 * np.mean(in_x & in_z)

    # Wie viele Samples in Cube-Zone (z 0.04-0.09)?
    in_cube_z = (samples_raw[:, IZ] >= 0.04) & (samples_raw[:, IZ] <= 0.09)
    pct_cube = 100 * np.mean(in_x & in_cube_z)

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Z (m) — Höhe', fontsize=12)
    ax.set_title(f'sigma_scale_z = {zs:.0f}\n'
                 f'σ_z(norm) = {zs:.0f}  →  σ_z(phys) = {phys_sigma_z:.3f} m',
                 fontsize=13, color=color)
    ax.set_xlim(-0.2, 1.0)
    ax.set_ylim(-0.25, 0.40)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='upper left')

    textstr = (f'σ_z(phys) = {phys_sigma_z*100:.1f} cm\n'
               f'In Bounds: {pct_xz:.1f}%\n'
               f'In Cube-Zone: {pct_cube:.1f}%')
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
p1 = os.path.join(os.path.dirname(__file__), "cem_z_scale_comparison_side.png")
plt.savefig(p1, dpi=150, bbox_inches='tight')
print(f"Gespeichert: {p1}")


# ============================================================================
# PLOT: Z-Marginalverteilung für alle 3 Scales überlagert
# ============================================================================

fig2, ax2 = plt.subplots(figsize=(10, 6))
fig2.suptitle("Z-Marginalverteilung (Weltkoordinaten) — sigma_scale_z Vergleich",
              fontsize=14, fontweight='bold')

mu_z_val = ACTION_MEAN[IZ]
std_z_val = ACTION_STD[IZ]

x_range = np.linspace(-0.3, 0.4, 500)

for zs, color in zip(Z_SCALES, COLORS):
    sigma_phys = VAR_SCALE * zs * std_z_val
    pdf = norm.pdf(x_range, loc=mu_z_val, scale=sigma_phys)
    ax2.fill_between(x_range, pdf, alpha=0.15, color=color)
    ax2.plot(x_range, pdf, color=color, linewidth=2.5,
             label=f'z_scale={zs:.0f}  (σ={sigma_phys*100:.1f} cm)')

# Bounds
ax2.axvline(WS_LOWER[2], color='red', linewidth=2, linestyle='--', label='Bounds [0.00, 0.12]')
ax2.axvline(WS_UPPER[2], color='red', linewidth=2, linestyle='--')
ax2.axvspan(WS_LOWER[2], WS_UPPER[2], color='red', alpha=0.05)

# Tisch + Cube
ax2.axvline(0.04, color='saddlebrown', linewidth=1.5, linestyle=':', alpha=0.7, label='Tisch (0.04)')
ax2.axvspan(0.04, 0.09, color='green', alpha=0.08, label='Cube-Zone')

# Mittelwert
ax2.axvline(mu_z_val, color='black', linewidth=1, linestyle=':', alpha=0.5, label=f'μ_z = {mu_z_val:.3f}')

ax2.set_xlabel('Z (m) — Höhe', fontsize=12)
ax2.set_ylabel('Dichte', fontsize=12)
ax2.legend(fontsize=10, loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-0.25, 0.35)

# Annotation: Anteil in Bounds für jede Scale
for zs, color in zip(Z_SCALES, COLORS):
    sigma_phys = VAR_SCALE * zs * std_z_val
    p_in = norm.cdf(WS_UPPER[2], mu_z_val, sigma_phys) - norm.cdf(WS_LOWER[2], mu_z_val, sigma_phys)
    p_cube = norm.cdf(0.09, mu_z_val, sigma_phys) - norm.cdf(0.04, mu_z_val, sigma_phys)
    ax2.annotate(f'z={zs:.0f}: {p_in*100:.0f}% in bounds, {p_cube*100:.0f}% in cube',
                 xy=(0.02, 0.97 - Z_SCALES.index(zs)*0.05),
                 xycoords='axes fraction', fontsize=9, color=color,
                 fontweight='bold')

plt.tight_layout()
p2 = os.path.join(os.path.dirname(__file__), "cem_z_scale_comparison_1d.png")
plt.savefig(p2, dpi=150, bbox_inches='tight')
print(f"Gespeichert: {p2}")

plt.close('all')
print("\nFertig!")
