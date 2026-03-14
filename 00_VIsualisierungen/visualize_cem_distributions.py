"""
Visualisierung der CEM-Normalverteilungen über dem Workspace.

Zeigt:
  1. Draufsicht (XY-Ebene): Wie die CEM-Samples im Arbeitsraum verteilt sind
  2. Seitenansicht (XZ-Ebene): Wie sigma_scale die z-Exploration beeinflusst
  
Für jede Ansicht: OHNE und MIT sigma_scale nebeneinander.
Action Bounds als rotes Rechteck.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
from scipy.stats import norm
import os

from masterarbeit_style import apply_style, TEXTWIDTH_IN, FONT_SIZE, figsize, save_ma_figure
apply_style()

# ============================================================================
# KONFIGURATION (aus plan_franka.yaml & typischen Dataset-Statistiken)
# ============================================================================

# Typische action_mean/std aus dem 1000-Episoden-Dataset (8D)
# [x_s, y_s, z_s, g_s, x_e, y_e, z_e, g_e]
# Wir betrachten nur die End-Effektor-Zielposition: x_e, y_e, z_e = indices 4,5,6
ACTION_MEAN_8D = np.array([0.400, 0.025, 0.050, 0.50, 0.400, 0.025, 0.050, 0.50])
ACTION_STD_8D  = np.array([0.121, 0.158, 0.070, 0.50, 0.121, 0.158, 0.070, 0.50])

# Sigma-Scale aus plan_franka.yaml
SIGMA_SCALE_XY = 1.0
SIGMA_SCALE_Z  = 3.0
VAR_SCALE = 1.0  # CEM var_scale

# Workspace Bounds aus plan_franka.yaml
WS_LOWER = np.array([0.05, -0.40, 0.00])  # x, y, z
WS_UPPER = np.array([0.80,  0.45, 0.12])  # x, y, z

# Indices für die Endposition (8D Action → x_e=4, y_e=5, z_e=6)
IX, IY, IZ = 4, 5, 6

# ============================================================================
# SAMPLE GENERIERUNG (wie CEM es macht)
# ============================================================================

N_SAMPLES = 3000
np.random.seed(42)

# --- CEM arbeitet im normalisierten Raum ---
# mu_norm = 0 (= Mittelwert der Trainingsdaten)
# sigma_norm = var_scale * sigma_scale (= 1.0 * [1,1,3,...] ODER [1,1,1,...])

# OHNE Sigma-Scale
sigma_norm_without = VAR_SCALE * np.ones(8)

# MIT Sigma-Scale
sigma_norm_with = VAR_SCALE * np.array([
    SIGMA_SCALE_XY, SIGMA_SCALE_XY, SIGMA_SCALE_Z, 1.0,
    SIGMA_SCALE_XY, SIGMA_SCALE_XY, SIGMA_SCALE_Z, 1.0
])

# Samples im normalisierten Raum (mu=0)
samples_norm_without = np.random.randn(N_SAMPLES, 8) * sigma_norm_without
samples_norm_with    = np.random.randn(N_SAMPLES, 8) * sigma_norm_with

# Zurück in Weltkoordinaten: world = norm * std + mean
samples_world_without = samples_norm_without * ACTION_STD_8D + ACTION_MEAN_8D
samples_world_with    = samples_norm_with    * ACTION_STD_8D + ACTION_MEAN_8D


# ============================================================================
# CLAMPING (Action Bounds)
# ============================================================================

def clamp_samples(samples, lower, upper, indices):
    """Clampt samples auf Workspace Bounds."""
    clamped = samples.copy()
    for i, idx in enumerate(indices):
        clamped[:, idx] = np.clip(clamped[:, idx], lower[i], upper[i])
    return clamped

samples_clamped_without = clamp_samples(samples_world_without, WS_LOWER, WS_UPPER, [IX, IY, IZ])
samples_clamped_with    = clamp_samples(samples_world_with,    WS_LOWER, WS_UPPER, [IX, IY, IZ])


# ============================================================================
# HILFSFUNKTIONEN
# ============================================================================

def gauss_1d(x, mu, sigma):
    """1D Gauss-Dichte."""
    return norm.pdf(x, loc=mu, scale=sigma)

def draw_confidence_ellipse(ax, mu_x, mu_y, sigma_x, sigma_y, n_std=2.0, **kwargs):
    """Zeichnet Konfidenz-Ellipse (2σ)."""
    ellipse = Ellipse(
        xy=(mu_x, mu_y),
        width=2 * n_std * sigma_x,
        height=2 * n_std * sigma_y,
        **kwargs
    )
    ax.add_patch(ellipse)
    return ellipse


# ============================================================================
# PLOT 1: DRAUFSICHT (XY-Ebene)
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.45))
fig.suptitle("CEM Sampling-Verteilung — Draufsicht (XY-Ebene, z integriert)", 
             fontweight='bold')

for ax_idx, (title, samples_raw, samples_clamp, sigma_norm) in enumerate([
    ("OHNE Sigma-Scale\n(σ = var_scale × 1.0 für alle)", 
     samples_world_without, samples_clamped_without, sigma_norm_without),
    ("MIT Sigma-Scale\n(σ_z = var_scale × 3.0)", 
     samples_world_with, samples_clamped_with, sigma_norm_with),
]):
    ax = axes[ax_idx]
    
    # Physische Standardabweichung in Weltkoordinaten
    phys_sigma_x = sigma_norm[IX] * ACTION_STD_8D[IX]
    phys_sigma_y = sigma_norm[IY] * ACTION_STD_8D[IY]
    mu_x = ACTION_MEAN_8D[IX]
    mu_y = ACTION_MEAN_8D[IY]
    
    # Workspace Bounds Rechteck
    ws_rect = patches.Rectangle(
        (WS_LOWER[0], WS_LOWER[1]), 
        WS_UPPER[0] - WS_LOWER[0], 
        WS_UPPER[1] - WS_LOWER[1],
        linewidth=2.5, edgecolor='red', facecolor='red', alpha=0.08,
        label=f'Action Bounds\n[{WS_LOWER[0]:.2f}–{WS_UPPER[0]:.2f}] × [{WS_LOWER[1]:.2f}–{WS_UPPER[1]:.2f}]'
    )
    ax.add_patch(ws_rect)
    
    # Scatter: Samples (vor Clamping) → grau, transparent
    ax.scatter(samples_raw[:, IX], samples_raw[:, IY], 
               s=3, alpha=0.15, c='gray', label='Samples (vor Clamping)')
    
    # Scatter: Samples (nach Clamping) → blau
    ax.scatter(samples_clamp[:, IX], samples_clamp[:, IY], 
               s=5, alpha=0.3, c='tab:blue', label='Samples (nach Clamping)')
    
    # Konfidenz-Ellipsen (1σ, 2σ, 3σ)
    for n_std, alpha in [(1, 0.5), (2, 0.3), (3, 0.15)]:
        draw_confidence_ellipse(
            ax, mu_x, mu_y, phys_sigma_x, phys_sigma_y,
            n_std=n_std, facecolor='none', edgecolor='darkblue',
            linewidth=1.5, linestyle='--', alpha=alpha,
            label=f'{n_std}σ Ellipse' if alpha==0.5 else None
        )
    
    # Mittelwert markieren
    ax.plot(mu_x, mu_y, 'k+', markersize=15, markeredgewidth=2, label=f'μ = ({mu_x:.3f}, {mu_y:.3f})')
    
    # Typische Würfel-Position
    ax.plot(0.40, 0.0, 'rs', markersize=10, markeredgewidth=2, 
            markerfacecolor='none', label='Typische Cube-Position')
    
    ax.set_xlabel('X (m) — vorwärts/rückwärts')
    ax.set_ylabel('Y (m) — links/rechts')
    ax.set_title(title)
    ax.set_xlim(-0.3, 1.1)
    ax.set_ylim(-0.8, 0.8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    
    # Statistik-Box
    textstr = (f'σ_x(phys) = {phys_sigma_x:.4f} m\n'
               f'σ_y(phys) = {phys_sigma_y:.4f} m\n'
               f'σ_x(norm) = {sigma_norm[IX]:.1f}\n'
               f'σ_y(norm) = {sigma_norm[IY]:.1f}')
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
save_ma_figure(fig, os.path.join(os.path.dirname(__file__), "cem_distribution_top_view"))


# ============================================================================
# PLOT 2: SEITENANSICHT (XZ-Ebene)
# ============================================================================

fig2, axes2 = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.45))
fig2.suptitle("CEM Sampling-Verteilung — Seitenansicht (XZ-Ebene, y integriert)", 
              fontweight='bold')

for ax_idx, (title, samples_raw, samples_clamp, sigma_norm) in enumerate([
    ("OHNE Sigma-Scale\n(σ_z_norm = 1.0 → σ_z_phys = 0.070 m)", 
     samples_world_without, samples_clamped_without, sigma_norm_without),
    ("MIT Sigma-Scale (z × 3.0)\n(σ_z_norm = 3.0 → σ_z_phys = 0.210 m)", 
     samples_world_with, samples_clamped_with, sigma_norm_with),
]):
    ax = axes2[ax_idx]
    
    phys_sigma_x = sigma_norm[IX] * ACTION_STD_8D[IX]
    phys_sigma_z = sigma_norm[IZ] * ACTION_STD_8D[IZ]
    mu_x = ACTION_MEAN_8D[IX]
    mu_z = ACTION_MEAN_8D[IZ]
    
    # Workspace Bounds Rechteck
    ws_rect = patches.Rectangle(
        (WS_LOWER[0], WS_LOWER[2]), 
        WS_UPPER[0] - WS_LOWER[0], 
        WS_UPPER[2] - WS_LOWER[2],
        linewidth=2.5, edgecolor='red', facecolor='red', alpha=0.08,
        label=f'Action Bounds\nz: [{WS_LOWER[2]:.2f}–{WS_UPPER[2]:.2f}] m'
    )
    ax.add_patch(ws_rect)
    
    # Tischplatte
    ax.axhline(y=0.04, color='saddlebrown', linewidth=3, alpha=0.6, linestyle='-', 
               label='Tischoberfläche (~0.04 m)')
    ax.axhspan(0, 0.04, color='saddlebrown', alpha=0.1)
    
    # Würfel-Höhe
    ax.axhspan(0.04, 0.09, color='green', alpha=0.08, label='Cube-Zone (0.04–0.09 m)')
    
    # Scatter: Samples (vor Clamping) → grau
    ax.scatter(samples_raw[:, IX], samples_raw[:, IZ], 
               s=3, alpha=0.15, c='gray', label='Samples (vor Clamping)')
    
    # Scatter: Samples (nach Clamping) → blau
    ax.scatter(samples_clamp[:, IX], samples_clamp[:, IZ], 
               s=5, alpha=0.3, c='tab:blue', label='Samples (nach Clamping)')
    
    # Konfidenz-Ellipsen (1σ, 2σ, 3σ)
    for n_std, alpha in [(1, 0.5), (2, 0.3), (3, 0.15)]:
        draw_confidence_ellipse(
            ax, mu_x, mu_z, phys_sigma_x, phys_sigma_z,
            n_std=n_std, facecolor='none', edgecolor='darkblue',
            linewidth=1.5, linestyle='--', alpha=alpha,
            label=f'{n_std}σ Ellipse' if alpha==0.5 else None
        )
    
    # Mittelwert
    ax.plot(mu_x, mu_z, 'k+', markersize=15, markeredgewidth=2, 
            label=f'μ = ({mu_x:.3f}, {mu_z:.3f})')
    
    ax.set_xlabel('X (m) — vorwärts/rückwärts')
    ax.set_ylabel('Z (m) — Höhe')
    ax.set_title(title)
    ax.set_xlim(-0.3, 1.1)
    ax.set_ylim(-0.3, 0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    
    # In-bounds Anteil berechnen
    in_x = (samples_raw[:, IX] >= WS_LOWER[0]) & (samples_raw[:, IX] <= WS_UPPER[0])
    in_z = (samples_raw[:, IZ] >= WS_LOWER[2]) & (samples_raw[:, IZ] <= WS_UPPER[2])
    pct_in = 100 * np.mean(in_x & in_z)
    
    textstr = (f'σ_x(phys) = {phys_sigma_x:.4f} m\n'
               f'σ_z(phys) = {phys_sigma_z:.4f} m\n'
               f'σ_x(norm) = {sigma_norm[IX]:.1f}\n'
               f'σ_z(norm) = {sigma_norm[IZ]:.1f}\n'
               f'In Bounds: {pct_in:.1f}%')
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
save_ma_figure(fig2, os.path.join(os.path.dirname(__file__), "cem_distribution_side_view"))


# ============================================================================
# PLOT 3: 1D Marginalverteilungen (x, y, z) mit/ohne Sigma-Scale
# ============================================================================

fig3, axes3 = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.32))
fig3.suptitle("CEM 1D-Marginalverteilungen in Weltkoordinaten (vor Clamping)", 
              fontweight='bold')

for col, (dim_name, dim_idx, ws_lo, ws_hi) in enumerate([
    ('X', IX, WS_LOWER[0], WS_UPPER[0]),
    ('Y', IY, WS_LOWER[1], WS_UPPER[1]),
    ('Z', IZ, WS_LOWER[2], WS_UPPER[2]),
]):
    ax = axes3[col]
    mu = ACTION_MEAN_8D[dim_idx]
    std_base = ACTION_STD_8D[dim_idx]
    
    # Sigma ohne/mit Scale
    sigma_without = VAR_SCALE * 1.0 * std_base
    sigma_with_scale = sigma_norm_with[dim_idx]
    sigma_with = VAR_SCALE * sigma_with_scale * std_base
    
    x_range = np.linspace(mu - 4*max(sigma_without, sigma_with), 
                          mu + 4*max(sigma_without, sigma_with), 500)
    
    pdf_without = gauss_1d(x_range, mu, sigma_without)
    pdf_with    = gauss_1d(x_range, mu, sigma_with)
    
    ax.fill_between(x_range, pdf_without, alpha=0.2, color='tab:orange')
    ax.plot(x_range, pdf_without, 'tab:orange', linewidth=2, 
            label=f'ohne Scale (σ={sigma_without:.4f})')
    
    ax.fill_between(x_range, pdf_with, alpha=0.2, color='tab:blue')
    ax.plot(x_range, pdf_with, 'tab:blue', linewidth=2, 
            label=f'mit Scale (σ={sigma_with:.4f})')
    
    # Action Bounds
    ax.axvline(ws_lo, color='red', linewidth=2, linestyle='--', label=f'Bounds [{ws_lo:.2f}, {ws_hi:.2f}]')
    ax.axvline(ws_hi, color='red', linewidth=2, linestyle='--')
    ax.axvspan(ws_lo, ws_hi, color='red', alpha=0.05)
    
    # Mittelwert
    ax.axvline(mu, color='black', linewidth=1, linestyle=':', alpha=0.5)
    
    ax.set_xlabel(f'{dim_name} (m)')
    ax.set_ylabel('Dichte')
    ax.set_title(f'{dim_name}-Dimension\n(action_std = {std_base:.4f}, scale = {sigma_with_scale:.1f}×)')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
save_ma_figure(fig3, os.path.join(os.path.dirname(__file__), "cem_distribution_1d_marginals"))

plt.close('all')
print("\nFertig!")
