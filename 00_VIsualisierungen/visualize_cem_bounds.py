"""
Action-Bounds-Wirkung: Zeigt den Unterschied zwischen rohem Gauß-Sampling
und bounds-konformem Truncated-Gaussian-Sampling.

Zwei separate Figuren (jeweils mit/ohne Beschriftung):
    1. XY-Ebene: Samples vor und nach Trunkierung, Bounds als Rechteck
    2. XZ-Seitenansicht: Samples vor und nach Trunkierung
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import norm
import os

from masterarbeit_style import apply_style
apply_style()

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cem_individual")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# PARAMETER
# ============================================================================

ACTION_MEAN = np.array([0.400, 0.025, 0.050, 0.50, 0.400, 0.025, 0.050, 0.50])
ACTION_STD  = np.array([0.121, 0.158, 0.070, 0.50, 0.121, 0.158, 0.070, 0.50])
VAR_SCALE = 1.0
IX, IY, IZ = 4, 5, 6

WS_LOWER = np.array([0.05, -0.40, 0.00])
WS_UPPER = np.array([0.80,  0.45, 0.12])

SIGMA_SCALE_Z = 3.0

N_SAMPLES = 3000
np.random.seed(42)

# Farben (mattes Blau)
BLUE_RAW       = '#B0C4D8'   # Blassblau für Samples VOR Trunkierung
BLUE_TRUNCATED = '#5A8FB4'   # Kräftigeres Blau für Samples NACH Trunkierung
BLUE_LINE    = '#6E96AE'
BLUE_FILL    = '#9DBDD5'
RED_BOUNDS   = '#C62828'

# ============================================================================
# SAMPLES GENERIEREN (Iteration 1, breite Verteilung)
# ============================================================================

mu_x, mu_y, mu_z = ACTION_MEAN[IX], ACTION_MEAN[IY], ACTION_MEAN[IZ]
sx = VAR_SCALE * ACTION_STD[IX]
sy = VAR_SCALE * ACTION_STD[IY]
sz = VAR_SCALE * SIGMA_SCALE_Z * ACTION_STD[IZ]  # mit sigma_scale z×3

raw_x = np.random.normal(mu_x, sx, N_SAMPLES)
raw_y = np.random.normal(mu_y, sy, N_SAMPLES)
raw_z = np.random.normal(mu_z, sz, N_SAMPLES)


def sample_truncated(raw_samples, lower, upper, mu, sigma, max_resample_iters=64):
    """Erzeugt bounds-konforme Samples mit Rejection Sampling.

    Die Anzahl der zurückgegebenen Samples bleibt exakt erhalten.
    """
    truncated = raw_samples.copy()
    invalid = (truncated < lower) | (truncated > upper)
    for _ in range(max_resample_iters):
        if not np.any(invalid):
            return truncated
        truncated[invalid] = np.random.normal(mu, sigma, invalid.sum())
        invalid = (truncated < lower) | (truncated > upper)

    # Fallback nur für pathologische Fälle; sollte praktisch nicht greifen.
    truncated = np.clip(truncated, lower, upper)
    return truncated


trunc_x = sample_truncated(raw_x, WS_LOWER[0], WS_UPPER[0], mu_x, sx)
trunc_y = sample_truncated(raw_y, WS_LOWER[1], WS_UPPER[1], mu_y, sy)
trunc_z = sample_truncated(raw_z, WS_LOWER[2], WS_UPPER[2], mu_z, sz)


def make_axes_black(ax):
    for spine in ax.spines.values():
        spine.set_color('black')
    ax.tick_params(colors='black', which='both')


def gauss(x, mu, sigma):
    return norm.pdf(x, loc=mu, scale=sigma)


# ============================================================================
# FIGUR 1: XY-EBENE — Einzelbild vor ODER nach Trunkierung
# ============================================================================

def create_xy_bounds_single(is_truncated, labeled=True):
    fig_size = 4.5
    fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size))

    samples_x = trunc_x if is_truncated else raw_x
    samples_y = trunc_y if is_truncated else raw_y
    dot_color = BLUE_TRUNCATED if is_truncated else BLUE_RAW
    title_text = 'Nach Trunkierung (Action Bounds)' if is_truncated else 'Vor Trunkierung (Gaußverteilung)'

    # Workspace Bounds
    rect = patches.Rectangle(
        (WS_LOWER[0], WS_LOWER[1]),
        WS_UPPER[0] - WS_LOWER[0], WS_UPPER[1] - WS_LOWER[1],
        linewidth=2.0, edgecolor=RED_BOUNDS, facecolor=RED_BOUNDS,
        alpha=0.06, linestyle='-', zorder=1
    )
    ax.add_patch(rect)

    # Bounds-Linien
    for val in [WS_LOWER[0], WS_UPPER[0]]:
        ax.axvline(val, color=RED_BOUNDS, lw=1.5, ls='--', alpha=0.7, zorder=2)
    for val in [WS_LOWER[1], WS_UPPER[1]]:
        ax.axhline(val, color=RED_BOUNDS, lw=1.5, ls='--', alpha=0.7, zorder=2)

    # Samples
    ax.scatter(samples_x, samples_y, s=4, alpha=0.4, c=dot_color,
               edgecolors='none', zorder=3)

    # Mittelwert
    ax.plot(mu_x, mu_y, 'k+', markersize=12, markeredgewidth=2, zorder=5)

    ax.set_xlim(-0.15, 1.0)
    ax.set_ylim(-0.65, 0.65)
    ax.set_aspect('equal')
    make_axes_black(ax)

    if labeled:
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(title_text, fontweight='bold', pad=8)
        ax.grid(True, alpha=0.15)

        ax.text(WS_UPPER[0] + 0.02, WS_LOWER[1] - 0.03,
                f'Bounds:\nX [{WS_LOWER[0]:.2f}, {WS_UPPER[0]:.2f}]\nY [{WS_LOWER[1]:.2f}, {WS_UPPER[1]:.2f}]',
                fontsize=8, color=RED_BOUNDS, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

        if is_truncated:
            ax.annotate('', xy=(WS_UPPER[0], 0.0), xytext=(WS_UPPER[0] + 0.12, 0.0),
                        arrowprops=dict(arrowstyle='->', color=RED_BOUNDS, lw=2))
            ax.annotate('', xy=(WS_LOWER[0], 0.0), xytext=(WS_LOWER[0] - 0.07, 0.0),
                        arrowprops=dict(arrowstyle='->', color=RED_BOUNDS, lw=2))
            ax.annotate('', xy=(0.4, WS_UPPER[1]), xytext=(0.4, WS_UPPER[1] + 0.12),
                        arrowprops=dict(arrowstyle='->', color=RED_BOUNDS, lw=2))
            ax.annotate('', xy=(0.4, WS_LOWER[1]), xytext=(0.4, WS_LOWER[1] - 0.12),
                        arrowprops=dict(arrowstyle='->', color=RED_BOUNDS, lw=2))
            in_bounds = ((raw_x >= WS_LOWER[0]) & (raw_x <= WS_UPPER[0]) &
                         (raw_y >= WS_LOWER[1]) & (raw_y <= WS_UPPER[1]))
            pct = 100 * np.mean(in_bounds)
            ax.text(0.03, 0.03, f'{pct:.0f}% der Samples\nlagen bereits in Bounds',
                    transform=ax.transAxes, fontsize=8, va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
        else:
            outside = ~((raw_x >= WS_LOWER[0]) & (raw_x <= WS_UPPER[0]) &
                        (raw_y >= WS_LOWER[1]) & (raw_y <= WS_UPPER[1]))
            pct_out = 100 * np.mean(outside)
            ax.text(0.03, 0.03, f'{pct_out:.0f}% der Samples\naußerhalb der Bounds',
                    transform=ax.transAxes, fontsize=8, va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(length=0)
        ax.grid(False)

    plt.tight_layout()
    return fig


# ============================================================================
# FIGUR 2: XZ-SEITENANSICHT — Einzelbild vor ODER nach Trunkierung
# ============================================================================

def create_xz_bounds_single(is_truncated, labeled=True):
    fig_size = 4.5
    fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size))

    s_x = trunc_x if is_truncated else raw_x
    s_z = trunc_z if is_truncated else raw_z
    dot_color = BLUE_TRUNCATED if is_truncated else BLUE_RAW
    title_text = 'Nach Trunkierung (Action Bounds)' if is_truncated else 'Vor Trunkierung (σ_z × 3.0)'

    # Workspace Bounds Rechteck (XZ)
    rect = patches.Rectangle(
        (WS_LOWER[0], WS_LOWER[2]),
        WS_UPPER[0] - WS_LOWER[0], WS_UPPER[2] - WS_LOWER[2],
        linewidth=2.0, edgecolor=RED_BOUNDS, facecolor=RED_BOUNDS,
        alpha=0.06, linestyle='-', zorder=1
    )
    ax.add_patch(rect)

    # Bounds-Linien
    for val in [WS_LOWER[0], WS_UPPER[0]]:
        ax.axvline(val, color=RED_BOUNDS, lw=1.5, ls='--', alpha=0.7, zorder=2)
    ax.axhline(WS_LOWER[2], color=RED_BOUNDS, lw=1.5, ls='--', alpha=0.7, zorder=2)
    ax.axhline(WS_UPPER[2], color=RED_BOUNDS, lw=1.5, ls='--', alpha=0.7, zorder=2)

    # Tischoberfläche
    ax.axhspan(-0.35, 0.04, color='saddlebrown', alpha=0.06, zorder=0)
    ax.axhline(0.04, color='saddlebrown', lw=2, alpha=0.5, zorder=2)

    # Cube-Zone
    ax.axhspan(0.04, 0.09, color='#2E7D32', alpha=0.06, zorder=0)

    # Samples
    ax.scatter(s_x, s_z, s=4, alpha=0.4, c=dot_color,
               edgecolors='none', zorder=3)

    # Mittelwert
    ax.plot(mu_x, mu_z, 'k+', markersize=12, markeredgewidth=2, zorder=5)

    ax.set_xlim(-0.15, 1.0)
    ax.set_ylim(-0.35, 0.45)
    ax.set_aspect('equal')
    make_axes_black(ax)

    if labeled:
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m) — Höhe')
        ax.set_title(title_text, fontweight='bold', pad=8)
        ax.grid(True, alpha=0.15)

        ax.text(WS_UPPER[0] + 0.02, WS_UPPER[2] + 0.01,
                f'Bounds:\nX [{WS_LOWER[0]:.2f}, {WS_UPPER[0]:.2f}]\n'
                f'Z [{WS_LOWER[2]:.2f}, {WS_UPPER[2]:.2f}]',
                fontsize=8, color=RED_BOUNDS, va='bottom', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

        ax.text(0.20, 0.025, 'Tisch', fontsize=8, color='saddlebrown',
                ha='center', va='center', alpha=0.7)
        ax.text(0.20, 0.065, 'Cube', fontsize=8, color='#2E7D32',
                ha='center', va='center', alpha=0.7)

        if is_truncated:
            ax.annotate('', xy=(0.4, WS_UPPER[2]),
                        xytext=(0.4, WS_UPPER[2] + 0.15),
                        arrowprops=dict(arrowstyle='->', color=RED_BOUNDS, lw=2))
            ax.annotate('', xy=(0.4, WS_LOWER[2]),
                        xytext=(0.4, WS_LOWER[2] - 0.15),
                        arrowprops=dict(arrowstyle='->', color=RED_BOUNDS, lw=2))
            in_z = 100 * np.mean((raw_z >= WS_LOWER[2]) & (raw_z <= WS_UPPER[2]))
            ax.text(0.03, 0.97,
                    f'{in_z:.0f}% der Samples\nlagen bereits im z-Intervall',
                    transform=ax.transAxes, fontsize=8, va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
        else:
            below = np.mean(raw_z < WS_LOWER[2]) * 100
            above = np.mean(raw_z > WS_UPPER[2]) * 100
            ax.text(0.03, 0.97,
                    f'{below:.0f}% unter z_min\n{above:.0f}% über z_max',
                    transform=ax.transAxes, fontsize=8, va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(length=0)
        ax.grid(False)

    plt.tight_layout()
    return fig


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    plots = [
        ('action_bounds_xy_raw',     lambda lab: create_xy_bounds_single(False, lab)),
        ('action_bounds_xy_truncated', lambda lab: create_xy_bounds_single(True, lab)),
        ('action_bounds_xz_raw',     lambda lab: create_xz_bounds_single(False, lab)),
        ('action_bounds_xz_truncated', lambda lab: create_xz_bounds_single(True, lab)),
    ]

    count = 0
    for name, create_fn in plots:
        for labeled in [True, False]:
            suffix = '' if labeled else '_unlabeled'
            fname = f"{name}{suffix}.svg"
            fpath = os.path.join(OUTPUT_DIR, fname)
            fig = create_fn(labeled)
            fig.savefig(fpath, format='svg')
            plt.close(fig)
            label_str = 'beschriftet' if labeled else 'unbeschriftet'
            print(f"  ✓ {fname}  ({label_str})")
            count += 1

    print(f"\n{count} SVGs gespeichert in: {OUTPUT_DIR}")
