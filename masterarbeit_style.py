"""
Masterarbeit-Stilkonfiguration für alle Visualisierungen.

Vorgaben:
  - Schrift: Helvetica, 11 pt
  - Bildbreite: 152.3 mm (volle Textbreite)
  - Ausgabeformate: SVG + PDF
  - Zwei Varianten: beschriftet & unbeschriftet

Nutzung:
  from masterarbeit_style import apply_style, TEXTWIDTH_IN, save_ma_figure
"""

import matplotlib
import matplotlib.pyplot as plt
import os

# ============================================================================
# ABMESSUNGEN
# ============================================================================

TEXTWIDTH_MM = 152.3
TEXTWIDTH_IN = TEXTWIDTH_MM / 25.4  # ≈ 5.996 inches

# Halbe Textbreite (z.B. für 2-spaltige Layouts)
HALFWIDTH_IN = TEXTWIDTH_IN / 2.0

# ============================================================================
# SCHRIFT & RCPARAMS
# ============================================================================

# Helvetica-Fallback-Kette: Helvetica → Helvetica Neue → Arial → sans-serif
FONT_FAMILY = "sans-serif"
FONT_SANS_SERIF = ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"]
FONT_SIZE = 11  # pt

MA_RCPARAMS = {
    # Schrift
    "font.family": FONT_FAMILY,
    "font.sans-serif": FONT_SANS_SERIF,
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE - 2,  # 9 pt
    "xtick.labelsize": FONT_SIZE - 1,  # 10 pt
    "ytick.labelsize": FONT_SIZE - 1,
    # Linien
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.5,
    # Export
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    # SVG/PDF
    "svg.fonttype": "none",        # Schrift als Text (nicht Pfade)
    "pdf.fonttype": 42,            # TrueType in PDF
    # Hintergrund
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    # LaTeX-kompatibel
    "mathtext.fontset": "custom",
    "mathtext.rm": "Helvetica",
    "mathtext.it": "Helvetica:italic",
    "mathtext.bf": "Helvetica:bold",
}


def apply_style():
    """Wendet den Masterarbeit-Stil global an."""
    matplotlib.use("Agg")
    plt.rcParams.update(MA_RCPARAMS)


def figsize(width_fraction=1.0, aspect=0.618):
    """
    Berechnet (width, height) in Inches für Masterarbeit-Figuren.

    Args:
        width_fraction: Anteil der Textbreite (1.0 = voll, 0.5 = halb)
        aspect: Seitenverhältnis height/width (default: Goldener Schnitt)

    Returns:
        (width_in, height_in)
    """
    w = TEXTWIDTH_IN * width_fraction
    h = w * aspect
    return (w, h)


def figsize_fixed(width_fraction=1.0, height_in=None):
    """
    Figurengröße mit fester Höhe in Inches.

    Args:
        width_fraction: Anteil der Textbreite
        height_in: Fixe Höhe in Inches

    Returns:
        (width_in, height_in)
    """
    w = TEXTWIDTH_IN * width_fraction
    return (w, height_in or w * 0.618)


# ============================================================================
# SPEICHERFUNKTION: SVG + PDF, beschriftet & unbeschriftet
# ============================================================================

def save_ma_figure(fig, path_stem, labeled_elements=None, create_unlabeled=True,
                   fixed_canvas=False):
    """
    Speichert eine Figur als SVG + PDF in zwei Varianten:
      1. <path_stem>.svg / .pdf  — beschriftet (unverändert)
      2. <path_stem>_unlabeled.svg / .pdf  — unbeschriftet
         (suptitle + annotierte Textboxen entfernt, Achsen/Labels bleiben)

    Args:
        fig: matplotlib Figure
        path_stem: Pfad ohne Dateiendung, z.B. "output/cem_xy"
        labeled_elements: Optionale Liste von matplotlib Artists, die in der
                          unbeschrifteten Variante ausgeblendet werden sollen.
                          Falls None, werden automatisch suptitle + text-Annotationen
                          (mit bbox) entfernt.
        create_unlabeled: Falls False, wird nur die beschriftete Variante erstellt.
        fixed_canvas: Falls True, speichert mit fixer Figure-Canvas-Groesse statt
                      tight-bbox. Damit bleiben beschriftete und unbeschriftete
                      Variante exakt gleich breit.
    """
    os.makedirs(os.path.dirname(path_stem) or ".", exist_ok=True)
    save_kwargs = {"bbox_inches": None} if fixed_canvas else {"bbox_inches": "tight"}

    # --- Beschriftete Variante ---
    fig.savefig(path_stem + ".svg", **save_kwargs)
    print(f"  ✓ {os.path.basename(path_stem)}.svg (beschriftet)")

    if not create_unlabeled:
        return

    # --- Unbeschriftete Variante ---
    # Sammle Elemente, die ausgeblendet werden
    hidden = []

    if labeled_elements is not None:
        for el in labeled_elements:
            if el is not None and hasattr(el, "set_visible"):
                el.set_visible(False)
                hidden.append(el)
    else:
        # Automatisch: suptitle entfernen
        if fig._suptitle is not None:
            fig._suptitle.set_visible(False)
            hidden.append(fig._suptitle)

        # Text-Annotationen mit bbox (= Statistik-Boxen) entfernen
        for ax in fig.get_axes():
            # Titel der Subplots entfernen
            title_obj = ax.title
            if title_obj and title_obj.get_text():
                title_obj.set_visible(False)
                hidden.append(title_obj)

            # Annotierte Text-Boxen (mit bbox) entfernen
            for txt in ax.texts:
                bbox = txt.get_bbox_patch()
                if bbox is not None:
                    txt.set_visible(False)
                    hidden.append(txt)

            # Legenden entfernen
            legend = ax.get_legend()
            if legend is not None:
                legend.set_visible(False)
                hidden.append(legend)

    fig.savefig(path_stem + "_unlabeled.svg", **save_kwargs)
    print(f"  ✓ {os.path.basename(path_stem)}_unlabeled.svg (unbeschriftet)")

    # Sichtbarkeit wiederherstellen
    for el in hidden:
        el.set_visible(True)
