"""
CEM-Konvergenz: Individuelle quadratische SVG-Plots mit Marginalverteilungen.

Jede Iteration als separates Bild, mit und ohne Beschriftung.
Helles mattes Blau für Verteilungen, schwarze Achsen.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
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
IX, IY = 4, 5

WS_LOWER = np.array([0.05, -0.40, 0.00])
WS_UPPER = np.array([0.80,  0.45, 0.12])

EEF_HOME = np.array([0.40, 0.00])
GOAL_POS = np.array([0.40, 0.00])

# Helles mattes Blau (gedämpft, nicht knallig)
BLUE_FILL = '#9DBDD5'
BLUE_LINE = '#6E96AE'
BLUE_DOTS = '#5A8FB4'  # Etwas kräftiger für Punkte

# Gleichgroße Achsen-Bereiche → quadratisch mit aspect='equal'
AXIS_RANGE = 1.0  # 1 Meter in jeder Richtung
X_CENTER = (WS_LOWER[0] + WS_UPPER[0]) / 2  # 0.425
Y_CENTER = (WS_LOWER[1] + WS_UPPER[1]) / 2  # 0.025
XLIM = (X_CENTER - AXIS_RANGE / 2, X_CENTER + AXIS_RANGE / 2)
YLIM = (Y_CENTER - AXIS_RANGE / 2, Y_CENTER + AXIS_RANGE / 2)

N_SAMPLES = 2000

CEM_ITERATIONS = [
    {
        'name': 'iter01',
        'label': 'Iteration 1 (Start)',
        'seed': 42,
        'mu_xy': np.array([ACTION_MEAN[IX], ACTION_MEAN[IY]]),
        'sigma_xy': np.array([VAR_SCALE * ACTION_STD[IX],
                              VAR_SCALE * ACTION_STD[IY]]),
        'show_eef': True,
    },
    {
        'name': 'iter15',
        'label': 'Iteration 15 (Mitte)',
        'seed': 57,
        'mu_xy': np.array([0.41, -0.01]),
        'sigma_xy': np.array([0.04, 0.05]),
        'show_eef': False,
    },
    {
        'name': 'iter30',
        'label': 'Iteration 30 (Konvergiert)',
        'seed': 73,
        'mu_xy': np.array([0.40, 0.00]),
        'sigma_xy': np.array([0.012, 0.015]),
        'show_eef': False,
    },
]


# ============================================================================
# HILFSFUNKTIONEN
# ============================================================================

def gauss(x, mu, sigma):
    return norm.pdf(x, loc=mu, scale=sigma)


def make_axes_black(ax):
    """Setzt alle Achsen-Elemente auf Schwarz."""
    for spine in ax.spines.values():
        spine.set_color('black')
    ax.tick_params(colors='black', which='both')


# ============================================================================
# PLOT-FUNKTION
# ============================================================================

def create_cem_plot(step_data, labeled=True):
    """Erstellt einen einzelnen CEM-Plot mit quadratischem Hauptplot + Marginals."""

    mu_x, mu_y = step_data['mu_xy']
    sx, sy = step_data['sigma_xy']

    # Reproduzierbare Samples
    rng = np.random.RandomState(step_data['seed'])
    raw_x = rng.normal(mu_x, sx, N_SAMPLES)
    raw_y = rng.normal(mu_y, sy, N_SAMPLES)
    samp_x = np.clip(raw_x, WS_LOWER[0], WS_UPPER[0])
    samp_y = np.clip(raw_y, WS_LOWER[1], WS_UPPER[1])

    # ---- Figur & Achsen-Layout ----
    fig_size = 4.5  # Quadratische Figur (Inches)
    fig = plt.figure(figsize=(fig_size, fig_size))

    # Positionen in Figur-Koordinaten [left, bottom, width, height]
    if labeled:
        left, bottom = 0.14, 0.14
        avail = 0.80          # verfügbarer Platz
    else:
        left, bottom = 0.02, 0.02
        avail = 0.94

    main_size = avail * 0.79  # Quadratische Hauptfläche
    gap       = avail * 0.03  # Abstand
    marg_size = avail * 0.18  # Marginal-Größe

    ax_main  = fig.add_axes([left, bottom, main_size, main_size])
    ax_top   = fig.add_axes([left, bottom + main_size + gap, main_size, marg_size])
    ax_right = fig.add_axes([left + main_size + gap, bottom, marg_size, main_size])

    # ================================================================
    # HAUPTPLOT (XY-Scatter)
    # ================================================================

    # Workspace Bounds
    rect = patches.Rectangle(
        (WS_LOWER[0], WS_LOWER[1]),
        WS_UPPER[0] - WS_LOWER[0], WS_UPPER[1] - WS_LOWER[1],
        linewidth=1.2, edgecolor='#D32F2F', facecolor='#D32F2F',
        alpha=0.06, linestyle='--', zorder=1
    )
    ax_main.add_patch(rect)

    # CEM-Samples
    ax_main.scatter(samp_x, samp_y, s=6, alpha=0.5, c=BLUE_DOTS,
                    edgecolors='none', zorder=3, label=f'{N_SAMPLES} CEM-Samples')

    # 1σ Ellipse
    ell = Ellipse((mu_x, mu_y), 2 * sx, 2 * sy,
                  facecolor=BLUE_FILL, edgecolor=BLUE_LINE,
                  linewidth=1.5, linestyle='--', alpha=0.15,
                  zorder=4, label='1σ')
    ax_main.add_patch(ell)

    # Mittelwert, Ziel, EEF Home nur in beschrifteter Variante
    if labeled:
        ax_main.plot(mu_x, mu_y, 'k+', markersize=12, markeredgewidth=2, zorder=5)

        # Ziel (Cube)
        ax_main.plot(GOAL_POS[0], GOAL_POS[1], '*', color='#2E7D32',
                     markersize=14, markeredgewidth=1, markeredgecolor='#1B5E20',
                     zorder=6, label='Zielposition (Cube)')

        # EEF Home (nur bei Iteration 1)
        if step_data.get('show_eef'):
            ax_main.plot(EEF_HOME[0], EEF_HOME[1], '^', markersize=9,
                         markerfacecolor='#B3D4FC', markeredgecolor='#1565C0',
                         markeredgewidth=1.5, zorder=6, label='EEF Home')

    ax_main.set_xlim(XLIM)
    ax_main.set_ylim(YLIM)
    ax_main.set_aspect('equal')
    make_axes_black(ax_main)

    # ================================================================
    # TOP MARGINAL (X-Verteilung)
    # ================================================================

    x_vals = np.linspace(XLIM[0], XLIM[1], 500)
    pdf_x = gauss(x_vals, mu_x, sx)
    ax_top.fill_between(x_vals, pdf_x, alpha=0.3, color=BLUE_FILL)
    ax_top.plot(x_vals, pdf_x, color=BLUE_LINE, linewidth=1.5)
    ax_top.axvline(WS_LOWER[0], color='#D32F2F', lw=0.8, ls='--', alpha=0.5)
    ax_top.axvline(WS_UPPER[0], color='#D32F2F', lw=0.8, ls='--', alpha=0.5)
    ax_top.set_xlim(XLIM)
    ax_top.set_yticks([])
    ax_top.tick_params(labelbottom=False)
    make_axes_black(ax_top)

    # ================================================================
    # RIGHT MARGINAL (Y-Verteilung)
    # ================================================================

    y_vals = np.linspace(YLIM[0], YLIM[1], 500)
    pdf_y = gauss(y_vals, mu_y, sy)
    ax_right.fill_betweenx(y_vals, pdf_y, alpha=0.3, color=BLUE_FILL)
    ax_right.plot(pdf_y, y_vals, color=BLUE_LINE, linewidth=1.5)
    ax_right.axhline(WS_LOWER[1], color='#D32F2F', lw=0.8, ls='--', alpha=0.5)
    ax_right.axhline(WS_UPPER[1], color='#D32F2F', lw=0.8, ls='--', alpha=0.5)
    ax_right.set_ylim(YLIM)
    ax_right.set_xticks([])
    ax_right.tick_params(labelleft=False)
    make_axes_black(ax_right)

    # ================================================================
    # BESCHRIFTUNG (nur bei labeled=True)
    # ================================================================

    if labeled:
        ax_main.set_xlabel('X (m)')
        ax_main.set_ylabel('Y (m)')
        ax_main.grid(True, alpha=0.15)
        ax_top.set_ylabel('p(x)', fontsize=9)
        ax_right.set_xlabel('p(y)', fontsize=9)
        ax_top.set_title(step_data['label'], fontweight='bold', pad=6)

        textstr = (f'μ = ({mu_x:.3f}, {mu_y:.3f})\n'
                   f'σ_x = {sx:.4f} m\n'
                   f'σ_y = {sy:.4f} m')
        ax_main.text(0.03, 0.97, textstr, transform=ax_main.transAxes,
                     va='top', ha='left', fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.3',
                               facecolor='white', alpha=0.85))
        ax_main.legend(loc='lower right', framealpha=0.85, fontsize=8)
    else:
        # Komplett ohne Beschriftung: kein Text, keine Ticks, kein Grid
        for ax in [ax_main, ax_top, ax_right]:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(length=0)
        ax_main.grid(False)

    return fig


# ============================================================================
# MAIN: Alle Plots generieren
# ============================================================================

if __name__ == '__main__':
    for step_data in CEM_ITERATIONS:
        for labeled in [True, False]:
            suffix = '' if labeled else '_unlabeled'
            fname = f"cem_xy_{step_data['name']}{suffix}.svg"
            fpath = os.path.join(OUTPUT_DIR, fname)

            fig = create_cem_plot(step_data, labeled=labeled)
            fig.savefig(fpath, format='svg')
            plt.close(fig)
            label_str = 'beschriftet' if labeled else 'unbeschriftet'
            print(f"  ✓ {fname}  ({label_str})")

    print(f"\n{len(CEM_ITERATIONS) * 2} SVGs gespeichert in: {OUTPUT_DIR}")
