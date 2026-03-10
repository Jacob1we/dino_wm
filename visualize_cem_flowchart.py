"""
CEM-Ablaufdiagramm für Masterarbeit — Franka Cube Stacking mit DINO World Model.

Professionelles Flowchart des Cross-Entropy Method (CEM) Planungsprozesses
mit allen Franka-spezifischen Erweiterungen.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# ============================================================================
# STYLE
# ============================================================================

# Farben
C_INPUT    = '#E8F5E9'   # Grün — Eingaben
C_PROCESS  = '#E3F2FD'   # Blau — Verarbeitungsschritte
C_DECISION = '#FFF3E0'   # Orange — Entscheidungen
C_OUTPUT   = '#FCE4EC'   # Rot/Pink — Ausgaben
C_LOOP     = '#F3E5F5'   # Lila — Loop-Körper
C_WM       = '#E0F7FA'   # Cyan — World Model
C_FRANKA   = '#FFF9C4'   # Gelb — Franka-Erweiterungen
C_BORDER   = '#37474F'
C_ARROW    = '#455A64'
C_LOOP_BG  = '#F5F5F5'

FONT_MAIN = 'DejaVu Sans'
FONT_SIZE_TITLE = 11
FONT_SIZE_BOX = 8.5
FONT_SIZE_SMALL = 7
FONT_SIZE_PARAM = 6.5

def draw_box(ax, x, y, w, h, text, color, border_color=C_BORDER,
             fontsize=FONT_SIZE_BOX, style='round', lw=1.5, 
             text_color='black', sub_text=None, sub_fontsize=FONT_SIZE_SMALL):
    """Zeichnet eine Box mit Text."""
    if style == 'round':
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="round,pad=0.08", 
                             facecolor=color, edgecolor=border_color,
                             linewidth=lw, zorder=3)
    elif style == 'diamond':
        # Raute als Polygon
        diamond = plt.Polygon(
            [(x, y + h/2), (x + w/2, y), (x, y - h/2), (x - w/2, y)],
            facecolor=color, edgecolor=border_color, linewidth=lw, zorder=3
        )
        ax.add_patch(diamond)
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                fontfamily=FONT_MAIN, fontweight='bold', color=text_color, zorder=4)
        return
    else:
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="square,pad=0.05",
                             facecolor=color, edgecolor=border_color,
                             linewidth=lw, zorder=3)
    ax.add_patch(box)
    
    if sub_text:
        ax.text(x, y + 0.15, text, ha='center', va='center', fontsize=fontsize,
                fontfamily=FONT_MAIN, fontweight='bold', color=text_color, zorder=4)
        ax.text(x, y - 0.25, sub_text, ha='center', va='center', fontsize=sub_fontsize,
                fontfamily=FONT_MAIN, color='#555555', zorder=4, style='italic')
    else:
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                fontfamily=FONT_MAIN, fontweight='bold', color=text_color, zorder=4)

def draw_arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.5, style='->', 
               label=None, label_fontsize=FONT_SIZE_SMALL):
    """Zeichnet einen Pfeil."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                               connectionstyle='arc3,rad=0'),
                zorder=2)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.15, my, label, fontsize=label_fontsize,
                fontfamily=FONT_MAIN, color='#666666', ha='left', va='center')

def draw_curved_arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.5, rad=0.3,
                      label=None, label_pos='right'):
    """Zeichnet einen gebogenen Pfeil."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                               connectionstyle=f'arc3,rad={rad}'),
                zorder=2)
    if label:
        mx = max(x1, x2) + 0.3 if label_pos == 'right' else min(x1, x2) - 0.3
        my = (y1 + y2) / 2
        ha = 'left' if label_pos == 'right' else 'right'
        ax.text(mx, my, label, fontsize=FONT_SIZE_SMALL, fontfamily=FONT_MAIN,
                color='#666666', ha=ha, va='center')


# ============================================================================
# DIAGRAMM
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 16))
ax.set_xlim(-5, 5)
ax.set_ylim(-17.5, 2)
ax.set_aspect('equal')
ax.axis('off')

# Titel
ax.text(0, 1.5, 'CEM-Planungsprozess', ha='center', va='center',
        fontsize=15, fontfamily=FONT_MAIN, fontweight='bold')
ax.text(0, 1.0, 'Cross-Entropy Method für Franka Cube Stacking mit DINO World Model',
        ha='center', va='center', fontsize=9, fontfamily=FONT_MAIN, color='#666666')

# --- Positionen (y verläuft von oben nach unten) ---
bw = 3.8   # Box-Breite
bh = 0.7   # Box-Höhe
bh_s = 0.55  # kleine Box
x0 = 0     # Zentrum

# ===================== EINGABEN =====================

y = 0
draw_box(ax, -2.2, y, 2.2, bh, 'Beobachtung obs₀', C_INPUT,
         sub_text='RGB 224×224 + EEF xyz')
draw_box(ax,  2.2, y, 2.2, bh, 'Zielbild obs_g', C_INPUT,
         sub_text='RGB 224×224 + EEF xyz')

# Pfeile nach unten
draw_arrow(ax, -2.2, y - bh/2, -2.2, y - 1.1)
draw_arrow(ax,  2.2, y - bh/2,  2.2, y - 1.1)

# ===================== PREPROCESSING =====================

y = -1.4
draw_box(ax, -2.2, y, 2.2, bh, 'Preprocessing', C_PROCESS,
         sub_text='Resize, Normalize, DINOv2')
draw_box(ax,  2.2, y, 2.2, bh, 'Encode Goal', C_WM,
         sub_text='z_g = WM.encode(obs_g)')

draw_arrow(ax, -2.2, y - bh/2, -2.2, y - 1.1)
draw_arrow(ax,  2.2, y - bh/2,  2.2, y - 3.85)  # direkt zum Loss

# ===================== INITIALISIERUNG =====================

y = -2.8
draw_box(ax, -2.2, y, 3.0, 0.85, 'Initialisierung', C_FRANKA,
         sub_text='μ = 0,  σ = var_scale × σ_scale')

# Parameter-Annotation rechts
ax.text(1.1, y, 
        'σ_scale = [1, 1, 3, 1, 1, 1, 3, 1]\n'
        'Gripper-Dims: σ = 0 (Mask)',
        fontsize=FONT_SIZE_PARAM, fontfamily=FONT_MAIN, color='#888888',
        va='center', ha='left',
        bbox=dict(boxstyle='round,pad=0.3', facecolor=C_FRANKA, 
                  edgecolor='#FBC02D', alpha=0.7, linewidth=0.8))

draw_arrow(ax, -2.2, y - 0.85/2, -2.2, y - 1.25)

# ===================== LOOP-RAHMEN =====================

loop_top = -4.2
loop_bot = -15.8
loop_left = -4.6
loop_right = 4.6

# Loop-Hintergrund
loop_bg = FancyBboxPatch((loop_left, loop_bot), loop_right - loop_left, loop_top - loop_bot,
                          boxstyle="round,pad=0.15", facecolor=C_LOOP_BG,
                          edgecolor='#7E57C2', linewidth=2.0, linestyle='--',
                          zorder=1, alpha=0.5)
ax.add_patch(loop_bg)

ax.text(loop_left + 0.2, loop_top - 0.15, 
        'Optimierungs-Loop  (i = 1 ... 30)',
        fontsize=9, fontfamily=FONT_MAIN, fontweight='bold', color='#7E57C2',
        va='top', ha='left')

# ===================== SAMPLING (im Loop) =====================

y = -4.9
draw_box(ax, x0, y, bw, bh, 'Sampling', C_LOOP,
         sub_text='aₖ ~ N(μ, σ²)  →  (300, H=5, D=8)')

# Parameter rechts
ax.text(2.4, y,
        '300 Samples\n'
        'Horizon = 5\n'
        'Action-Dim = 8',
        fontsize=FONT_SIZE_PARAM, fontfamily=FONT_MAIN, color='#888888',
        va='center', ha='left',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='#BDBDBD', alpha=0.7, linewidth=0.5))

draw_arrow(ax, x0, y - bh/2, x0, y - 1.0)

# ===================== CLAMP =====================

y = -6.2
draw_box(ax, x0, y, bw, bh, 'Action Bounds Clamping', C_FRANKA,
         sub_text='x∈[0.05, 0.80]  y∈[−0.40, 0.45]  z∈[0.00, 0.12] m')

draw_arrow(ax, x0, y - bh/2, x0, y - 1.0)

# ===================== GRIPPER QUANTIZE =====================

y = -7.5
draw_box(ax, x0, y, bw, bh, 'Gripper-Quantisierung', C_FRANKA,
         sub_text='g ∈ {0, 1}  (binär, Mask: σ_g = 0)')

draw_arrow(ax, x0, y - bh/2, x0, y - 1.0)

# ===================== WORLD MODEL ROLLOUT =====================

y = -8.85
draw_box(ax, x0, y, bw, 0.9, 'World Model Rollout', C_WM,
         sub_text='ẑ₁...ẑ_H = WM.rollout(obs₀, a₁...a_H)')

# WM Details links
ax.text(-2.5, y,
        'DINOv2 Encoder\n'
        '+ ViT Predictor\n'
        '+ Proprio Encoder',
        fontsize=FONT_SIZE_PARAM, fontfamily=FONT_MAIN, color='#888888',
        va='center', ha='right',
        bbox=dict(boxstyle='round,pad=0.3', facecolor=C_WM,
                  edgecolor='#00ACC1', alpha=0.7, linewidth=0.8))

draw_arrow(ax, x0, y - 0.9/2, x0, y - 1.25)

# ===================== LOSS =====================

y = -10.4
draw_box(ax, x0, y, bw, 0.85, 'Objective (Loss)', C_PROCESS,
         sub_text='L = MSE(ẑ_H, z_g)_visual + α · MSE(ẑ_H, z_g)_proprio')

# z_g Pfeil von oben rechts einzeichnen
draw_curved_arrow(ax, 2.2, -1.4 - bh/2 - 0.3, 1.9, y + 0.85/2,
                  color='#00897B', lw=1.2, rad=-0.15, label='z_g', label_pos='right')

# Parameter
ax.text(-2.5, y,
        'α = 0.5\n'
        'mode = last',
        fontsize=FONT_SIZE_PARAM, fontfamily=FONT_MAIN, color='#888888',
        va='center', ha='right',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='#BDBDBD', alpha=0.7, linewidth=0.5))

draw_arrow(ax, x0, y - 0.85/2, x0, y - 1.25)

# ===================== TOP-K SELEKTION =====================

y = -12.0
draw_box(ax, x0, y, bw, bh, 'Top-K Selektion', C_LOOP,
         sub_text='K=30 Samples mit niedrigstem Loss')

draw_arrow(ax, x0, y - bh/2, x0, y - 1.0)

# ===================== UPDATE MU, SIGMA =====================

y = -13.35
draw_box(ax, x0, y, bw, 0.85, 'Update μ, σ', C_LOOP,
         sub_text='μ ← mean(Top-K),  σ ← std(Top-K)')

# Nachbearbeitung
ax.text(2.5, y,
        'μ auf Bounds\nclampen\n'
        'σ_gripper = 0',
        fontsize=FONT_SIZE_PARAM, fontfamily=FONT_MAIN, color='#888888',
        va='center', ha='left',
        bbox=dict(boxstyle='round,pad=0.3', facecolor=C_FRANKA,
                  edgecolor='#FBC02D', alpha=0.7, linewidth=0.8))

draw_arrow(ax, x0, y - 0.85/2, x0, y - 1.15)

# ===================== ENTSCHEIDUNG =====================

y = -14.9
draw_box(ax, x0, y, 2.8, 0.85, 'i < opt_steps?', C_DECISION, style='round',
         fontsize=FONT_SIZE_BOX)

# Ja-Pfeil: zurück nach oben zum Sampling
draw_curved_arrow(ax, -1.4, y, -3.7, -4.9,
                  color='#43A047', lw=2.0, rad=-0.35,
                  label='  Ja', label_pos='left')
ax.text(-4.3, -9.8, 'Ja →\nweitere\nIteration',
        fontsize=FONT_SIZE_SMALL, fontfamily=FONT_MAIN, color='#43A047',
        ha='center', va='center', fontweight='bold')

# Nein-Pfeil: nach unten
draw_arrow(ax, x0, y - 0.85/2, x0, y - 1.3, label='Nein', color='#E53935')

# ===================== AUSGABE =====================

y = -16.55
draw_box(ax, x0, y, bw, 0.8, 'Ausgabe: μ* (beste Aktion)', C_OUTPUT,
         sub_text='a* = [x_s, y_s, z_s, g_s, x_e, y_e, z_e, g_e]')

# ===================== MPC-Hinweis =====================

ax.text(0, -17.2,
        'In MPC: Erste Sub-Action a*[0] ausführen → neues Bild → erneut planen',
        fontsize=FONT_SIZE_SMALL, fontfamily=FONT_MAIN, color='#7E57C2',
        ha='center', va='center', style='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#F3E5F5',
                  edgecolor='#7E57C2', alpha=0.5, linewidth=0.8))

# ===================== LEGENDE =====================

legend_x = 3.0
legend_y = -0.5
legend_items = [
    (C_INPUT,   'Eingabe'),
    (C_PROCESS, 'Verarbeitung'),
    (C_WM,      'World Model'),
    (C_FRANKA,  'Franka-Erweiterung'),
    (C_LOOP,    'CEM-Loop'),
    (C_DECISION,'Entscheidung'),
    (C_OUTPUT,  'Ausgabe'),
]
for i, (color, label) in enumerate(legend_items):
    yy = legend_y - i * 0.35
    rect = FancyBboxPatch((legend_x - 0.2, yy - 0.12), 0.35, 0.24,
                          boxstyle="round,pad=0.03", facecolor=color,
                          edgecolor=C_BORDER, linewidth=0.8)
    ax.add_patch(rect)
    ax.text(legend_x + 0.35, yy, label, fontsize=FONT_SIZE_SMALL,
            fontfamily=FONT_MAIN, va='center', ha='left')

plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), "cem_flowchart.png")
plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Gespeichert: {output_path}")

# Auch als PDF für LaTeX
output_pdf = os.path.join(os.path.dirname(__file__), "cem_flowchart.pdf")
plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')
print(f"Gespeichert: {output_pdf}")

plt.close('all')
print("Fertig!")
