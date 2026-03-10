"""
Masterarbeit-Abbildungen: CEM-Optimierung über Iterationen, MPC-Timesteps & Sigma-Scales.

Erstellt 6 Figuren:
  1. XY-Draufsicht: 3 CEM-Iterationen (i=1, i=15, i=30) mit Gaußglocken
  2. Z-Seitenansicht: 3 CEM-Iterationen mit Gaußglocken
  3. XY-Punktwolken: 300 Samples bei σ_scale = 1, 2, 3
  4. Z-Punktwolken: 300 Samples bei σ_scale = 1, 2, 3
  5. XY-Draufsicht: 3 MPC-Timesteps (EEF bewegt sich), jeweils 15 CEM-Iterationen
  6. Z-Seitenansicht: 3 MPC-Timesteps (EEF senkt sich), jeweils 15 CEM-Iterationen
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse, FancyBboxPatch
from matplotlib.gridspec import GridSpec
from scipy.stats import norm
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# PARAMETER
# ============================================================================

# Dataset-Statistiken (8D)
ACTION_MEAN = np.array([0.400, 0.025, 0.050, 0.50, 0.400, 0.025, 0.050, 0.50])
ACTION_STD  = np.array([0.121, 0.158, 0.070, 0.50, 0.121, 0.158, 0.070, 0.50])
VAR_SCALE = 1.0
IX, IY, IZ = 4, 5, 6  # Indices für x_e, y_e, z_e in 8D

# Workspace Bounds
WS_LOWER = np.array([0.05, -0.40, 0.00])
WS_UPPER = np.array([0.80,  0.45, 0.12])

# Positionen
EEF_HOME  = np.array([0.40, 0.00, 0.42])   # Home-Position des EEF
CUBE_POS  = np.array([0.40, 0.00, 0.065])  # Würfel auf Tisch
GOAL_POS  = np.array([0.40, 0.00, 0.065])  # Ziel = Cube greifen

# Simulated CEM convergence over 30 optimization steps
# At each "timestep" we show: μ, σ, and samples
# t=0: initial (broad), t=15: converging, t=30: converged

SIGMA_SCALE_Z = 3.0  # aktuelle Config

np.random.seed(42)

# --- Simulierte CEM-Konvergenz (3 Iterationen) ---
# Wir simulieren wie μ und σ sich über opt_steps verändern
CEM_ITERATIONS = [
    {
        'label': 'Iteration 1 (Start)',
        'step': 1,
        'mu_xy': np.array([ACTION_MEAN[IX], ACTION_MEAN[IY]]),  # μ = action_mean
        'sigma_xy': np.array([VAR_SCALE * ACTION_STD[IX], VAR_SCALE * ACTION_STD[IY]]),
        'mu_z': ACTION_MEAN[IZ],
        'sigma_z': VAR_SCALE * SIGMA_SCALE_Z * ACTION_STD[IZ],
        'color': '#E57373',  # Rot
        'alpha': 0.4,
    },
    {
        'label': 'Iteration 15 (Mitte)',
        'step': 15,
        'mu_xy': np.array([0.41, -0.01]),  # μ verschiebt sich Richtung Ziel
        'sigma_xy': np.array([0.04, 0.05]),  # σ schrumpft
        'mu_z': 0.06,
        'sigma_z': 0.03,
        'color': '#FFB74D',  # Orange
        'alpha': 0.5,
    },
    {
        'label': 'Iteration 30 (Konvergiert)',
        'step': 30,
        'mu_xy': np.array([0.40, 0.00]),  # μ ≈ Zielposition
        'sigma_xy': np.array([0.012, 0.015]),  # σ sehr klein
        'mu_z': 0.065,
        'sigma_z': 0.008,
        'color': '#4CAF50',  # Grün
        'alpha': 0.7,
    },
]

# --- MPC-Timesteps: EEF bewegt sich, CEM hat jeweils 15 Iters konvergiert ---
# t=0: EEF bei Home, plant Richtung Cube
# t=1: EEF hat sich in XY Richtung Cube bewegt
# t=2: EEF über dem Cube, senkt sich ab
MPC_TIMESTEPS = [
    {
        'label': 'Timestep t=0 (Start)',
        'eef_xy': np.array([0.30, 0.15]),     # EEF startet versetzt
        'eef_z': 0.10,                         # Höhe: oben
        'mu_xy': np.array([0.35, 0.08]),       # CEM μ nach 15 Iters → Richtung Cube
        'sigma_xy': np.array([0.035, 0.04]),   # σ nach 15 Iters (konvergiert)
        'mu_z': 0.07,
        'sigma_z': 0.015,
        'color': '#5C6BC0',  # Indigo
    },
    {
        'label': 'Timestep t=1 (Annäherung)',
        'eef_xy': np.array([0.37, 0.05]),      # EEF näher am Cube
        'eef_z': 0.08,                          # Tiefer
        'mu_xy': np.array([0.39, 0.02]),        # CEM μ → fast am Cube
        'sigma_xy': np.array([0.025, 0.03]),
        'mu_z': 0.065,
        'sigma_z': 0.012,
        'color': '#26A69A',  # Teal
    },
    {
        'label': 'Timestep t=2 (Greifen)',
        'eef_xy': np.array([0.40, 0.01]),      # EEF über Cube
        'eef_z': 0.065,                         # Cube-Höhe
        'mu_xy': np.array([0.40, 0.00]),        # CEM μ = Cube exakt
        'sigma_xy': np.array([0.015, 0.018]),
        'mu_z': 0.065,
        'sigma_z': 0.008,
        'color': '#EF5350',  # Rot
    },
]


# ============================================================================
# HILFSFUNKTIONEN
# ============================================================================

def gauss(x, mu, sigma):
    return norm.pdf(x, loc=mu, scale=sigma)

def draw_ws_bounds_xy(ax):
    """Zeichnet Workspace Bounds als rotes Rechteck (XY)."""
    rect = patches.Rectangle(
        (WS_LOWER[0], WS_LOWER[1]),
        WS_UPPER[0] - WS_LOWER[0],
        WS_UPPER[1] - WS_LOWER[1],
        linewidth=2, edgecolor='#D32F2F', facecolor='#D32F2F', alpha=0.04,
        linestyle='-', zorder=1
    )
    ax.add_patch(rect)

def draw_ws_bounds_xz(ax):
    """Zeichnet Workspace Bounds als rotes Rechteck (XZ)."""
    rect = patches.Rectangle(
        (WS_LOWER[0], WS_LOWER[2]),
        WS_UPPER[0] - WS_LOWER[0],
        WS_UPPER[2] - WS_LOWER[2],
        linewidth=2, edgecolor='#D32F2F', facecolor='#D32F2F', alpha=0.04,
        linestyle='-', zorder=1
    )
    ax.add_patch(rect)


# ============================================================================
# FIGURE 1: XY-DRAUFSICHT — 3 CEM-Iterationen mit Gaußglocken
# ============================================================================

def create_fig1_xy_iterations():
    fig = plt.figure(figsize=(20, 7.5))
    
    # GridSpec: 3 Hauptplots mit marginal distributions
    # Jeder Subplot bekommt einen Haupt-Scatter + X-Marginal oben + Y-Marginal rechts
    outer_gs = GridSpec(1, 3, figure=fig, wspace=0.35)
    
    for col, step_data in enumerate(CEM_ITERATIONS):
        # Inner GridSpec für Haupt + Marginals
        inner_gs = outer_gs[col].subgridspec(
            2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
            hspace=0.05, wspace=0.05
        )
        
        ax_main = fig.add_subplot(inner_gs[1, 0])
        ax_top  = fig.add_subplot(inner_gs[0, 0], sharex=ax_main)
        ax_right = fig.add_subplot(inner_gs[1, 1], sharey=ax_main)
        
        mu_x, mu_y = step_data['mu_xy']
        sx, sy = step_data['sigma_xy']
        color = step_data['color']
        
        # --- Samples generieren ---
        samples_x = np.random.normal(mu_x, sx, 300)
        samples_y = np.random.normal(mu_y, sy, 300)
        # Clamping
        samples_x_c = np.clip(samples_x, WS_LOWER[0], WS_UPPER[0])
        samples_y_c = np.clip(samples_y, WS_LOWER[1], WS_UPPER[1])
        
        # === MAIN PLOT (XY Scatter) ===
        draw_ws_bounds_xy(ax_main)
        
        # Samples
        ax_main.scatter(samples_x_c, samples_y_c, s=8, alpha=0.35, c=color,
                       edgecolors='none', zorder=3, label='300 CEM-Samples')
        
        # Konfidenz-Ellipsen
        for n_std, a in [(1, 0.6), (2, 0.35), (3, 0.15)]:
            e = Ellipse((mu_x, mu_y), 2*n_std*sx, 2*n_std*sy,
                       facecolor='none', edgecolor=color, linewidth=1.8,
                       linestyle='--', alpha=a, zorder=4,
                       label=f'{n_std}σ' if n_std == 1 else None)
            ax_main.add_patch(e)
        
        # μ markieren
        ax_main.plot(mu_x, mu_y, 'k+', markersize=14, markeredgewidth=2.5, zorder=5)
        ax_main.plot(mu_x, mu_y, marker='o', markersize=4, color=color, zorder=5)
        
        # Ziel markieren
        ax_main.plot(GOAL_POS[0], GOAL_POS[1], 'g*', markersize=16, 
                    markeredgewidth=1, markeredgecolor='darkgreen', zorder=6,
                    label='Zielposition (Cube)')
        
        # EEF Home
        if col == 0:
            ax_main.plot(EEF_HOME[0], EEF_HOME[1], 'b^', markersize=10,
                        markeredgewidth=1.5, markerfacecolor='lightblue',
                        markeredgecolor='blue', zorder=6, label='EEF Home')
        
        ax_main.set_xlim(WS_LOWER[0] - 0.08, WS_UPPER[0] + 0.08)
        ax_main.set_ylim(WS_LOWER[1] - 0.12, WS_UPPER[1] + 0.12)
        ax_main.set_xlabel('X (m)', fontsize=10)
        if col == 0:
            ax_main.set_ylabel('Y (m)', fontsize=10)
        ax_main.set_aspect('equal')
        ax_main.grid(True, alpha=0.2)
        ax_main.legend(fontsize=6.5, loc='lower right', framealpha=0.8)
        
        # Statistik
        textstr = (f'μ = ({mu_x:.3f}, {mu_y:.3f})\n'
                   f'σ_x = {sx:.4f} m\n'
                   f'σ_y = {sy:.4f} m')
        ax_main.text(0.02, 0.98, textstr, transform=ax_main.transAxes, fontsize=7,
                    va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
        
        # === TOP MARGINAL (X-Gaußglocke) ===
        x_range = np.linspace(WS_LOWER[0] - 0.08, WS_UPPER[0] + 0.08, 300)
        pdf_x = gauss(x_range, mu_x, sx)
        ax_top.fill_between(x_range, pdf_x, alpha=0.25, color=color)
        ax_top.plot(x_range, pdf_x, color=color, linewidth=1.8)
        # Bounds
        ax_top.axvline(WS_LOWER[0], color='#D32F2F', linewidth=1, linestyle='--', alpha=0.6)
        ax_top.axvline(WS_UPPER[0], color='#D32F2F', linewidth=1, linestyle='--', alpha=0.6)
        ax_top.set_ylabel('p(x)', fontsize=7)
        ax_top.tick_params(labelbottom=False, labelsize=6)
        ax_top.set_title(step_data['label'], fontsize=11, fontweight='bold', color=color, pad=8)
        ax_top.set_yticks([])
        
        # === RIGHT MARGINAL (Y-Gaußglocke) ===
        y_range = np.linspace(WS_LOWER[1] - 0.12, WS_UPPER[1] + 0.12, 300)
        pdf_y = gauss(y_range, mu_y, sy)
        ax_right.fill_betweenx(y_range, pdf_y, alpha=0.25, color=color)
        ax_right.plot(pdf_y, y_range, color=color, linewidth=1.8)
        # Bounds
        ax_right.axhline(WS_LOWER[1], color='#D32F2F', linewidth=1, linestyle='--', alpha=0.6)
        ax_right.axhline(WS_UPPER[1], color='#D32F2F', linewidth=1, linestyle='--', alpha=0.6)
        ax_right.set_xlabel('p(y)', fontsize=7)
        ax_right.tick_params(labelleft=False, labelsize=6)
        ax_right.set_xticks([])
    
    # Supertitle mit Loss-Erklärung
    fig.suptitle(
        'CEM-Konvergenz über 30 Iterationen — XY-Draufsicht mit Marginalverteilungen\n'
        r'$\mathcal{L} = \mathrm{MSE}(\hat{z}_H^{\mathrm{vis}},\, z_g^{\mathrm{vis}}) '
        r'+ \alpha \cdot \mathrm{MSE}(\hat{z}_H^{\mathrm{proprio}},\, z_g^{\mathrm{proprio}})$'
        r'      wobei $\hat{z}_H$ = DINO WM Rollout-Ausgang am Horizont $H$,  $z_g$ = enkodiertes Zielbild',
        fontsize=11, fontweight='bold', y=1.02
    )
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "ma_cem_xy_iterations.png")
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ {path}")


# ============================================================================
# FIGURE 2: Z-SEITENANSICHT — 3 CEM-Iterationen mit Gaußglocken
# ============================================================================

def create_fig2_z_iterations():
    fig = plt.figure(figsize=(20, 7))
    
    outer_gs = GridSpec(1, 3, figure=fig, wspace=0.35)
    
    for col, step_data in enumerate(CEM_ITERATIONS):
        # Inner GridSpec: Hauptplot + Z-Marginal rechts
        inner_gs = outer_gs[col].subgridspec(
            2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
            hspace=0.05, wspace=0.05
        )
        
        ax_main = fig.add_subplot(inner_gs[1, 0])
        ax_top  = fig.add_subplot(inner_gs[0, 0], sharex=ax_main)
        ax_right = fig.add_subplot(inner_gs[1, 1], sharey=ax_main)
        
        mu_x = step_data['mu_xy'][0]
        sx = step_data['sigma_xy'][0]
        mu_z = step_data['mu_z']
        sz = step_data['sigma_z']
        color = step_data['color']
        
        # Samples
        samples_x = np.random.normal(mu_x, sx, 300)
        samples_z = np.random.normal(mu_z, sz, 300)
        samples_x_c = np.clip(samples_x, WS_LOWER[0], WS_UPPER[0])
        samples_z_c = np.clip(samples_z, WS_LOWER[2], WS_UPPER[2])
        
        # === MAIN PLOT (XZ) ===
        draw_ws_bounds_xz(ax_main)
        
        # Tisch + Cube Zone
        ax_main.axhline(y=0.04, color='saddlebrown', linewidth=2.5, alpha=0.5, zorder=1)
        ax_main.axhspan(0, 0.04, color='saddlebrown', alpha=0.08, zorder=0,
                       label='Tisch')
        ax_main.axhspan(0.04, 0.09, color='#A5D6A7', alpha=0.12, zorder=0,
                       label='Cube-Zone')
        
        # Samples
        ax_main.scatter(samples_x_c, samples_z_c, s=8, alpha=0.35, c=color,
                       edgecolors='none', zorder=3, label='300 Samples')
        
        # Konfidenz-Ellipsen
        for n_std, a in [(1, 0.6), (2, 0.35), (3, 0.15)]:
            e = Ellipse((mu_x, mu_z), 2*n_std*sx, 2*n_std*sz,
                       facecolor='none', edgecolor=color, linewidth=1.8,
                       linestyle='--', alpha=a, zorder=4,
                       label=f'{n_std}σ' if n_std == 1 else None)
            ax_main.add_patch(e)
        
        # μ
        ax_main.plot(mu_x, mu_z, 'k+', markersize=14, markeredgewidth=2.5, zorder=5)
        
        # Cube-Höhe
        ax_main.axhline(y=CUBE_POS[2], color='green', linewidth=1, linestyle=':',
                       alpha=0.6, zorder=2)
        ax_main.plot(CUBE_POS[0], CUBE_POS[2], 'g*', markersize=14,
                    markeredgewidth=1, markeredgecolor='darkgreen', zorder=6,
                    label='Cube-Höhe')
        
        ax_main.set_xlim(WS_LOWER[0] - 0.08, WS_UPPER[0] + 0.08)
        ax_main.set_ylim(WS_LOWER[2], WS_UPPER[2])
        ax_main.set_xlabel('X (m)', fontsize=10)
        if col == 0:
            ax_main.set_ylabel('Z (m) — Höhe', fontsize=10)
        ax_main.grid(True, alpha=0.2)
        ax_main.legend(fontsize=6.5, loc='upper left', framealpha=0.8)
        
        # Statistik
        textstr = (f'μ_z = {mu_z:.4f} m\n'
                   f'σ_z = {sz:.4f} m\n'
                   f'σ_z = {sz*100:.1f} cm')
        ax_main.text(0.98, 0.98, textstr, transform=ax_main.transAxes, fontsize=7,
                    va='top', ha='right',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
        
        # === TOP MARGINAL (X) ===
        x_range = np.linspace(WS_LOWER[0] - 0.08, WS_UPPER[0] + 0.08, 300)
        pdf_x = gauss(x_range, mu_x, sx)
        ax_top.fill_between(x_range, pdf_x, alpha=0.25, color=color)
        ax_top.plot(x_range, pdf_x, color=color, linewidth=1.8)
        ax_top.axvline(WS_LOWER[0], color='#D32F2F', linewidth=1, linestyle='--', alpha=0.6)
        ax_top.axvline(WS_UPPER[0], color='#D32F2F', linewidth=1, linestyle='--', alpha=0.6)
        ax_top.tick_params(labelbottom=False, labelsize=6)
        ax_top.set_title(step_data['label'], fontsize=11, fontweight='bold', color=color, pad=8)
        ax_top.set_yticks([])
        
        # === RIGHT MARGINAL (Z-Gaußglocke, auf Bounds beschränkt) ===
        z_range = np.linspace(WS_LOWER[2], WS_UPPER[2], 300)
        pdf_z = gauss(z_range, mu_z, sz)
        ax_right.fill_betweenx(z_range, pdf_z, alpha=0.25, color=color)
        ax_right.plot(pdf_z, z_range, color=color, linewidth=1.8)
        # Bounds
        ax_right.axhline(WS_LOWER[2], color='#D32F2F', linewidth=1, linestyle='--', alpha=0.6)
        ax_right.axhline(WS_UPPER[2], color='#D32F2F', linewidth=1, linestyle='--', alpha=0.6)
        # Tisch
        ax_right.axhline(0.04, color='saddlebrown', linewidth=1, alpha=0.4)
        ax_right.tick_params(labelleft=False, labelsize=6)
        ax_right.set_xticks([])
    
    fig.suptitle(
        'CEM-Konvergenz über 30 Iterationen — Seitenansicht (XZ) mit Marginalverteilungen\n'
        r'$\hat{z}_H = f_{\theta}(z_0, a_1, \ldots, a_H)$ — DINO World Model propagiert '
        r'Latent-State $z_0$ über $H{=}5$ Schritte mit Aktionssequenz',
        fontsize=11, fontweight='bold', y=1.02
    )
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "ma_cem_z_iterations.png")
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ {path}")


# ============================================================================
# FIGURE 3: XY-PUNKTWOLKEN — 3 Sigma-Scales
# ============================================================================

def create_fig3_xy_sigma_scales():
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle(
        'CEM-Sampling bei verschiedenen Sigma-Scales — XY-Draufsicht (Iteration 1)\n'
        r'$a_k \sim \mathcal{N}(\mu,\, (\mathrm{var\_scale} \cdot \sigma_{\mathrm{scale}} '
        r'\cdot \sigma_{\mathrm{action}})^2)$'
        r'      300 Samples, $\mu = \bar{a}_{\mathrm{Dataset}}$',
        fontsize=12, fontweight='bold'
    )
    
    SCALES = [
        (1.0, 'tab:orange', r'$\sigma_{\mathrm{scale}}$ = 1.0 (uniform)'),
        (2.0, 'tab:green',  r'$\sigma_{\mathrm{scale}}$ = 2.0'),
        (3.0, 'tab:blue',   r'$\sigma_{\mathrm{scale}}$ = 3.0 (z-kompensiert)'),
    ]
    
    for i, (scale, color, title) in enumerate(SCALES):
        ax = axes[i]
        
        # Sampling (physisch: sigma_phys = var_scale * scale * action_std)
        sx = VAR_SCALE * scale * ACTION_STD[IX]
        sy = VAR_SCALE * scale * ACTION_STD[IY]
        mu_x, mu_y = ACTION_MEAN[IX], ACTION_MEAN[IY]
        
        samples_x = np.random.normal(mu_x, sx, 300)
        samples_y = np.random.normal(mu_y, sy, 300)
        samples_x_c = np.clip(samples_x, WS_LOWER[0], WS_UPPER[0])
        samples_y_c = np.clip(samples_y, WS_LOWER[1], WS_UPPER[1])
        
        # Workspace Bounds
        draw_ws_bounds_xy(ax)
        # Bounds-Labels an den Rändern
        ax.axvline(WS_LOWER[0], color='#D32F2F', linewidth=1.2, linestyle='--', alpha=0.5)
        ax.axvline(WS_UPPER[0], color='#D32F2F', linewidth=1.2, linestyle='--', alpha=0.5)
        ax.axhline(WS_LOWER[1], color='#D32F2F', linewidth=1.2, linestyle='--', alpha=0.5)
        ax.axhline(WS_UPPER[1], color='#D32F2F', linewidth=1.2, linestyle='--', alpha=0.5)
        
        # Vor-Clamping Samples (grau)
        ax.scatter(samples_x, samples_y, s=10, alpha=0.12, c='gray',
                  edgecolors='none', zorder=2, label='vor Clamping')
        
        # Nach-Clamping Samples
        ax.scatter(samples_x_c, samples_y_c, s=12, alpha=0.45, c=color,
                  edgecolors='none', zorder=3, label='nach Clamping')
        
        # Konfidenz-Ellipsen
        for n_std, a in [(1, 0.6), (2, 0.3), (3, 0.12)]:
            e = Ellipse((mu_x, mu_y), 2*n_std*sx, 2*n_std*sy,
                       facecolor='none', edgecolor=color, linewidth=2,
                       linestyle='--', alpha=a, zorder=4,
                       label=f'{n_std}σ' if n_std == 1 else None)
            ax.add_patch(e)
        
        # μ
        ax.plot(mu_x, mu_y, 'k+', markersize=14, markeredgewidth=2.5, zorder=5,
               label=f'μ = ({mu_x:.2f}, {mu_y:.2f})')
        
        # Cube
        ax.plot(CUBE_POS[0], CUBE_POS[1], 'g*', markersize=14,
               markeredgewidth=1, markeredgecolor='darkgreen', zorder=6)
        
        # In bounds %
        in_bounds = np.mean(
            (samples_x >= WS_LOWER[0]) & (samples_x <= WS_UPPER[0]) &
            (samples_y >= WS_LOWER[1]) & (samples_y <= WS_UPPER[1])
        ) * 100
        
        ax.set_xlim(-0.25, 1.05)
        ax.set_ylim(-0.75, 0.75)
        ax.set_xlabel('X (m)', fontsize=10)
        if i == 0:
            ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_title(title, fontsize=12, color=color, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=7, loc='lower right', framealpha=0.8)
        
        textstr = (f'σ_x = {sx:.3f} m\n'
                   f'σ_y = {sy:.3f} m\n'
                   f'In Bounds: {in_bounds:.0f}%')
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=8,
               va='top', ha='left',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "ma_cem_xy_sigma_scales.png")
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ {path}")


# ============================================================================
# FIGURE 4: Z-PUNKTWOLKEN — 3 Sigma-Scales
# ============================================================================

def create_fig4_z_sigma_scales():
    fig = plt.figure(figsize=(20, 7.5))
    
    SCALES = [
        (1.0, 'tab:orange', r'$\sigma_{\mathrm{scale,z}}$ = 1.0'),
        (2.0, 'tab:green',  r'$\sigma_{\mathrm{scale,z}}$ = 2.0'),
        (3.0, 'tab:blue',   r'$\sigma_{\mathrm{scale,z}}$ = 3.0'),
    ]
    
    outer_gs = GridSpec(1, 3, figure=fig, wspace=0.3)
    
    for col, (scale, color, title) in enumerate(SCALES):
        inner_gs = outer_gs[col].subgridspec(
            1, 2, width_ratios=[4, 1.3], wspace=0.05
        )
        
        ax_main = fig.add_subplot(inner_gs[0])
        ax_right = fig.add_subplot(inner_gs[1], sharey=ax_main)
        
        # Sampling
        sx = VAR_SCALE * 1.0 * ACTION_STD[IX]  # x bleibt scale=1
        sz = VAR_SCALE * scale * ACTION_STD[IZ]  # z variiert
        mu_x, mu_z = ACTION_MEAN[IX], ACTION_MEAN[IZ]
        
        samples_x = np.random.normal(mu_x, sx, 300)
        samples_z = np.random.normal(mu_z, sz, 300)
        samples_x_c = np.clip(samples_x, WS_LOWER[0], WS_UPPER[0])
        samples_z_c = np.clip(samples_z, WS_LOWER[2], WS_UPPER[2])
        
        # === MAIN PLOT ===
        draw_ws_bounds_xz(ax_main)
        
        # Tisch
        ax_main.axhline(y=0.04, color='saddlebrown', linewidth=2.5, alpha=0.5)
        ax_main.axhspan(0, 0.04, color='saddlebrown', alpha=0.08, label='Tisch')
        ax_main.axhspan(0.04, 0.09, color='#A5D6A7', alpha=0.12, label='Cube-Zone')
        
        # Bounds
        ax_main.axhline(WS_LOWER[2], color='#D32F2F', linewidth=1.2, linestyle='--', alpha=0.5)
        ax_main.axhline(WS_UPPER[2], color='#D32F2F', linewidth=1.2, linestyle='--', alpha=0.5)
        
        # Vor-Clamping (grau)
        ax_main.scatter(samples_x, samples_z, s=10, alpha=0.12, c='gray',
                       edgecolors='none', zorder=2, label='vor Clamping')
        
        # Nach-Clamping
        ax_main.scatter(samples_x_c, samples_z_c, s=12, alpha=0.45, c=color,
                       edgecolors='none', zorder=3, label='nach Clamping')
        
        # Ellipsen
        for n_std, a in [(1, 0.6), (2, 0.3), (3, 0.12)]:
            e = Ellipse((mu_x, mu_z), 2*n_std*sx, 2*n_std*sz,
                       facecolor='none', edgecolor=color, linewidth=2,
                       linestyle='--', alpha=a, zorder=4,
                       label=f'{n_std}σ' if n_std == 1 else None)
            ax_main.add_patch(e)
        
        # μ
        ax_main.plot(mu_x, mu_z, 'k+', markersize=14, markeredgewidth=2.5, zorder=5)
        
        # Cube
        ax_main.plot(CUBE_POS[0], CUBE_POS[2], 'g*', markersize=14,
                    markeredgewidth=1, markeredgecolor='darkgreen', zorder=6)
        
        # Stats
        in_z = np.mean((samples_z >= WS_LOWER[2]) & (samples_z <= WS_UPPER[2])) * 100
        in_cube = np.mean((samples_z >= 0.04) & (samples_z <= 0.09)) * 100
        
        ax_main.set_xlim(WS_LOWER[0] - 0.1, WS_UPPER[0] + 0.1)
        ax_main.set_ylim(WS_LOWER[2], WS_UPPER[2])
        ax_main.set_xlabel('X (m)', fontsize=10)
        if col == 0:
            ax_main.set_ylabel('Z (m) — Höhe', fontsize=10)
        ax_main.grid(True, alpha=0.2)
        ax_main.legend(fontsize=6.5, loc='upper left', framealpha=0.8)
        ax_main.set_title(title, fontsize=12, color=color, fontweight='bold')
        
        textstr = (f'σ_z = {sz*100:.1f} cm\n'
                   f'In z-Bounds: {in_z:.0f}%\n'
                   f'In Cube-Zone: {in_cube:.0f}%')
        ax_main.text(0.98, 0.98, textstr, transform=ax_main.transAxes, fontsize=8,
                    va='top', ha='right',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))
        
        # === RIGHT MARGINAL (Z-Gaußglocke, auf Bounds beschränkt) ===
        z_range = np.linspace(WS_LOWER[2], WS_UPPER[2], 400)
        pdf_z = gauss(z_range, mu_z, sz)
        pdf_z_norm = pdf_z / pdf_z.max() * 0.8  # Normiert für hübsche Darstellung
        
        ax_right.fill_betweenx(z_range, pdf_z_norm, alpha=0.25, color=color)
        ax_right.plot(pdf_z_norm, z_range, color=color, linewidth=2)
        
        # Bounds + Tisch im Marginal
        ax_right.axhline(WS_LOWER[2], color='#D32F2F', linewidth=1.2, linestyle='--', alpha=0.5)
        ax_right.axhline(WS_UPPER[2], color='#D32F2F', linewidth=1.2, linestyle='--', alpha=0.5)
        ax_right.axhline(0.04, color='saddlebrown', linewidth=1, alpha=0.4)
        ax_right.axhspan(0.04, 0.09, color='#A5D6A7', alpha=0.08)
        
        # μ markieren
        if WS_LOWER[2] <= mu_z <= WS_UPPER[2]:
            ax_right.axhline(mu_z, color='black', linewidth=0.8, linestyle=':', alpha=0.5)
        
        ax_right.tick_params(labelleft=False, labelsize=6)
        ax_right.set_xlabel('p(z)', fontsize=8)
        ax_right.set_xlim(0, 1.0)
        ax_right.grid(True, alpha=0.15)
    
    fig.suptitle(
        'CEM-Sampling bei verschiedenen z-Sigma-Scales — Seitenansicht (XZ)\n'
        r'$\sigma_{z,\mathrm{phys}} = \mathrm{var\_scale} \times \sigma_{\mathrm{scale,z}} '
        r'\times \sigma_{\mathrm{action,z}}$  mit  $\sigma_{\mathrm{action,z}} = 0.070\,\mathrm{m}$',
        fontsize=12, fontweight='bold', y=1.02
    )
    
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "ma_cem_z_sigma_scales.png")
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ {path}")


# ============================================================================
# FIGURE 5: XY-DRAUFSICHT — 3 MPC-TIMESTEPS (EEF bewegt sich)
# ============================================================================

def create_fig5_xy_timesteps():
    fig = plt.figure(figsize=(20, 7.5))
    outer_gs = GridSpec(1, 3, figure=fig, wspace=0.35)

    for col, ts in enumerate(MPC_TIMESTEPS):
        inner_gs = outer_gs[col].subgridspec(
            2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
            hspace=0.05, wspace=0.05
        )

        ax_main  = fig.add_subplot(inner_gs[1, 0])
        ax_top   = fig.add_subplot(inner_gs[0, 0], sharex=ax_main)
        ax_right = fig.add_subplot(inner_gs[1, 1], sharey=ax_main)

        mu_x, mu_y = ts['mu_xy']
        sx, sy = ts['sigma_xy']
        eef_x, eef_y = ts['eef_xy']
        color = ts['color']

        # Samples (nach 15 CEM-Iterationen)
        samples_x = np.random.normal(mu_x, sx, 300)
        samples_y = np.random.normal(mu_y, sy, 300)
        samples_x_c = np.clip(samples_x, WS_LOWER[0], WS_UPPER[0])
        samples_y_c = np.clip(samples_y, WS_LOWER[1], WS_UPPER[1])

        # === MAIN PLOT ===
        draw_ws_bounds_xy(ax_main)

        # Samples
        ax_main.scatter(samples_x_c, samples_y_c, s=8, alpha=0.35, c=color,
                        edgecolors='none', zorder=3, label='300 CEM-Samples')

        # Konfidenz-Ellipsen
        for n_std, a in [(1, 0.6), (2, 0.35), (3, 0.15)]:
            e = Ellipse((mu_x, mu_y), 2*n_std*sx, 2*n_std*sy,
                        facecolor='none', edgecolor=color, linewidth=1.8,
                        linestyle='--', alpha=a, zorder=4,
                        label=f'{n_std}σ' if n_std == 1 else None)
            ax_main.add_patch(e)

        # μ (CEM-Ergebnis)
        ax_main.plot(mu_x, mu_y, 'k+', markersize=14, markeredgewidth=2.5, zorder=5)
        ax_main.plot(mu_x, mu_y, marker='o', markersize=4, color=color, zorder=5)

        # EEF aktuelle Position
        ax_main.plot(eef_x, eef_y, 'b^', markersize=11,
                     markeredgewidth=1.5, markerfacecolor='lightblue',
                     markeredgecolor='blue', zorder=6,
                     label=f'EEF ({eef_x:.2f}, {eef_y:.2f})')

        # Pfeil EEF → μ
        ax_main.annotate('', xy=(mu_x, mu_y), xytext=(eef_x, eef_y),
                         arrowprops=dict(arrowstyle='->', color=color, lw=2,
                                         connectionstyle='arc3,rad=0.1'),
                         zorder=5)

        # Ziel (Cube)
        ax_main.plot(GOAL_POS[0], GOAL_POS[1], 'g*', markersize=16,
                     markeredgewidth=1, markeredgecolor='darkgreen', zorder=6,
                     label='Ziel (Cube)')

        # Trajektorie bisheriger EEF-Positionen
        if col > 0:
            prev_x = [MPC_TIMESTEPS[j]['eef_xy'][0] for j in range(col)]
            prev_y = [MPC_TIMESTEPS[j]['eef_xy'][1] for j in range(col)]
            ax_main.plot(prev_x + [eef_x], prev_y + [eef_y],
                         'b--', linewidth=1.2, alpha=0.4, zorder=2)
            ax_main.scatter(prev_x, prev_y, s=30, c='lightblue',
                            edgecolors='blue', linewidth=0.8, alpha=0.5, zorder=2)

        ax_main.set_xlim(WS_LOWER[0] - 0.08, WS_UPPER[0] + 0.08)
        ax_main.set_ylim(WS_LOWER[1] - 0.12, WS_UPPER[1] + 0.12)
        ax_main.set_xlabel('X (m)', fontsize=10)
        if col == 0:
            ax_main.set_ylabel('Y (m)', fontsize=10)
        ax_main.set_aspect('equal')
        ax_main.grid(True, alpha=0.2)
        ax_main.legend(fontsize=6.5, loc='lower right', framealpha=0.8)

        textstr = (f'EEF = ({eef_x:.2f}, {eef_y:.2f})\n'
                   f'μ = ({mu_x:.3f}, {mu_y:.3f})\n'
                   f'σ_x = {sx:.4f} m\n'
                   f'σ_y = {sy:.4f} m\n'
                   f'15 CEM-Iterationen')
        ax_main.text(0.02, 0.98, textstr, transform=ax_main.transAxes, fontsize=7,
                     va='top', ha='left',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

        # === TOP MARGINAL (X) ===
        x_range = np.linspace(WS_LOWER[0] - 0.08, WS_UPPER[0] + 0.08, 300)
        pdf_x = gauss(x_range, mu_x, sx)
        ax_top.fill_between(x_range, pdf_x, alpha=0.25, color=color)
        ax_top.plot(x_range, pdf_x, color=color, linewidth=1.8)
        ax_top.axvline(WS_LOWER[0], color='#D32F2F', linewidth=1, linestyle='--', alpha=0.6)
        ax_top.axvline(WS_UPPER[0], color='#D32F2F', linewidth=1, linestyle='--', alpha=0.6)
        ax_top.axvline(eef_x, color='blue', linewidth=1, linestyle=':', alpha=0.5)
        ax_top.set_ylabel('p(x)', fontsize=7)
        ax_top.tick_params(labelbottom=False, labelsize=6)
        ax_top.set_title(ts['label'], fontsize=11, fontweight='bold', color=color, pad=8)
        ax_top.set_yticks([])

        # === RIGHT MARGINAL (Y) ===
        y_range = np.linspace(WS_LOWER[1] - 0.12, WS_UPPER[1] + 0.12, 300)
        pdf_y = gauss(y_range, mu_y, sy)
        ax_right.fill_betweenx(y_range, pdf_y, alpha=0.25, color=color)
        ax_right.plot(pdf_y, y_range, color=color, linewidth=1.8)
        ax_right.axhline(WS_LOWER[1], color='#D32F2F', linewidth=1, linestyle='--', alpha=0.6)
        ax_right.axhline(WS_UPPER[1], color='#D32F2F', linewidth=1, linestyle='--', alpha=0.6)
        ax_right.axhline(eef_y, color='blue', linewidth=1, linestyle=':', alpha=0.5)
        ax_right.set_xlabel('p(y)', fontsize=7)
        ax_right.tick_params(labelleft=False, labelsize=6)
        ax_right.set_xticks([])

    fig.suptitle(
        'MPC-Planung über 3 Timesteps — XY-Draufsicht (je 15 CEM-Iterationen pro Step)\n'
        r'EEF bewegt sich $\rightarrow$ neues Bild $\rightarrow$ CEM re-plant ab aktuellem $z_0$',
        fontsize=11, fontweight='bold', y=1.02
    )

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "ma_cem_xy_timesteps.png")
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ {path}")


# ============================================================================
# FIGURE 6: Z-SEITENANSICHT — 3 MPC-TIMESTEPS (EEF senkt sich)
# ============================================================================

def create_fig6_z_timesteps():
    fig = plt.figure(figsize=(20, 7))
    outer_gs = GridSpec(1, 3, figure=fig, wspace=0.35)

    for col, ts in enumerate(MPC_TIMESTEPS):
        inner_gs = outer_gs[col].subgridspec(
            2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
            hspace=0.05, wspace=0.05
        )

        ax_main  = fig.add_subplot(inner_gs[1, 0])
        ax_top   = fig.add_subplot(inner_gs[0, 0], sharex=ax_main)
        ax_right = fig.add_subplot(inner_gs[1, 1], sharey=ax_main)

        mu_x = ts['mu_xy'][0]
        sx = ts['sigma_xy'][0]
        eef_x = ts['eef_xy'][0]
        eef_z = ts['eef_z']
        mu_z = ts['mu_z']
        sz = ts['sigma_z']
        color = ts['color']

        # Samples
        samples_x = np.random.normal(mu_x, sx, 300)
        samples_z = np.random.normal(mu_z, sz, 300)
        samples_x_c = np.clip(samples_x, WS_LOWER[0], WS_UPPER[0])
        samples_z_c = np.clip(samples_z, WS_LOWER[2], WS_UPPER[2])

        # === MAIN PLOT (XZ) ===
        draw_ws_bounds_xz(ax_main)

        # Tisch + Cube Zone
        ax_main.axhline(y=0.04, color='saddlebrown', linewidth=2.5, alpha=0.5, zorder=1)
        ax_main.axhspan(0, 0.04, color='saddlebrown', alpha=0.08, zorder=0, label='Tisch')
        ax_main.axhspan(0.04, 0.09, color='#A5D6A7', alpha=0.12, zorder=0, label='Cube-Zone')

        # Samples
        ax_main.scatter(samples_x_c, samples_z_c, s=8, alpha=0.35, c=color,
                        edgecolors='none', zorder=3, label='300 Samples')

        # Konfidenz-Ellipsen
        for n_std, a in [(1, 0.6), (2, 0.35), (3, 0.15)]:
            e = Ellipse((mu_x, mu_z), 2*n_std*sx, 2*n_std*sz,
                        facecolor='none', edgecolor=color, linewidth=1.8,
                        linestyle='--', alpha=a, zorder=4,
                        label=f'{n_std}σ' if n_std == 1 else None)
            ax_main.add_patch(e)

        # μ (CEM-Ergebnis)
        ax_main.plot(mu_x, mu_z, 'k+', markersize=14, markeredgewidth=2.5, zorder=5)

        # EEF aktuelle Position
        ax_main.plot(eef_x, eef_z, 'b^', markersize=11,
                     markeredgewidth=1.5, markerfacecolor='lightblue',
                     markeredgecolor='blue', zorder=6, label=f'EEF (z={eef_z:.3f})')

        # Pfeil EEF → μ
        ax_main.annotate('', xy=(mu_x, mu_z), xytext=(eef_x, eef_z),
                         arrowprops=dict(arrowstyle='->', color=color, lw=2,
                                         connectionstyle='arc3,rad=0.1'),
                         zorder=5)

        # Cube
        ax_main.plot(CUBE_POS[0], CUBE_POS[2], 'g*', markersize=14,
                     markeredgewidth=1, markeredgecolor='darkgreen', zorder=6,
                     label='Cube')

        # Trajektorie bisheriger EEFs
        if col > 0:
            prev_x = [MPC_TIMESTEPS[j]['eef_xy'][0] for j in range(col)]
            prev_z = [MPC_TIMESTEPS[j]['eef_z'] for j in range(col)]
            ax_main.plot(prev_x + [eef_x], prev_z + [eef_z],
                         'b--', linewidth=1.2, alpha=0.4, zorder=2)
            ax_main.scatter(prev_x, prev_z, s=30, c='lightblue',
                            edgecolors='blue', linewidth=0.8, alpha=0.5, zorder=2)

        ax_main.set_xlim(WS_LOWER[0] - 0.08, WS_UPPER[0] + 0.08)
        ax_main.set_ylim(WS_LOWER[2], WS_UPPER[2])
        ax_main.set_xlabel('X (m)', fontsize=10)
        if col == 0:
            ax_main.set_ylabel('Z (m) — Höhe', fontsize=10)
        ax_main.grid(True, alpha=0.2)
        ax_main.legend(fontsize=6.5, loc='upper left', framealpha=0.8)

        textstr = (f'EEF_z = {eef_z:.3f} m\n'
                   f'μ_z = {mu_z:.4f} m\n'
                   f'σ_z = {sz:.4f} m\n'
                   f'15 CEM-Iterationen')
        ax_main.text(0.98, 0.98, textstr, transform=ax_main.transAxes, fontsize=7,
                     va='top', ha='right',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

        # === TOP MARGINAL (X) ===
        x_range = np.linspace(WS_LOWER[0] - 0.08, WS_UPPER[0] + 0.08, 300)
        pdf_x = gauss(x_range, mu_x, sx)
        ax_top.fill_between(x_range, pdf_x, alpha=0.25, color=color)
        ax_top.plot(x_range, pdf_x, color=color, linewidth=1.8)
        ax_top.axvline(WS_LOWER[0], color='#D32F2F', linewidth=1, linestyle='--', alpha=0.6)
        ax_top.axvline(WS_UPPER[0], color='#D32F2F', linewidth=1, linestyle='--', alpha=0.6)
        ax_top.axvline(eef_x, color='blue', linewidth=1, linestyle=':', alpha=0.5)
        ax_top.tick_params(labelbottom=False, labelsize=6)
        ax_top.set_title(ts['label'], fontsize=11, fontweight='bold', color=color, pad=8)
        ax_top.set_yticks([])

        # === RIGHT MARGINAL (Z, auf Bounds beschränkt) ===
        z_range = np.linspace(WS_LOWER[2], WS_UPPER[2], 300)
        pdf_z = gauss(z_range, mu_z, sz)
        ax_right.fill_betweenx(z_range, pdf_z, alpha=0.25, color=color)
        ax_right.plot(pdf_z, z_range, color=color, linewidth=1.8)
        ax_right.axhline(WS_LOWER[2], color='#D32F2F', linewidth=1, linestyle='--', alpha=0.6)
        ax_right.axhline(WS_UPPER[2], color='#D32F2F', linewidth=1, linestyle='--', alpha=0.6)
        ax_right.axhline(0.04, color='saddlebrown', linewidth=1, alpha=0.4)
        ax_right.axhline(eef_z, color='blue', linewidth=1, linestyle=':', alpha=0.5)
        ax_right.tick_params(labelleft=False, labelsize=6)
        ax_right.set_xticks([])

    fig.suptitle(
        'MPC-Planung über 3 Timesteps — Seitenansicht (XZ), je 15 CEM-Iterationen\n'
        r'EEF senkt sich zum Cube $\downarrow$ — neues Bild pro Step $\rightarrow$ CEM re-plant',
        fontsize=11, fontweight='bold', y=1.02
    )

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "ma_cem_z_timesteps.png")
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ {path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("Generiere Masterarbeit-Abbildungen...")
    create_fig1_xy_iterations()
    create_fig2_z_iterations()
    create_fig3_xy_sigma_scales()
    create_fig4_z_sigma_scales()
    create_fig5_xy_timesteps()
    create_fig6_z_timesteps()
    print("\nAlle Abbildungen erstellt!")
