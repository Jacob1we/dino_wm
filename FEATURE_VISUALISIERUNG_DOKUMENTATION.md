# Feature-Visualisierung des DINO World Models

## Übersicht

Dieses Dokument beschreibt den vollständigen Ablauf, wie aus einem RGB-Eingabebild
die verschiedenen Feature-Visualisierungen des DINO World Models erzeugt werden.
Es dient als Referenz für die Masterarbeit und deckt alle fünf Visualisierungstypen
sowie die mathematischen Grundlagen der Multi-Head Self-Attention ab.

---

## Modellparameter

| Parameter | Wert | Beschreibung |
|-----------|------|--------------|
| Modell | `dinov2_vits14` | DINOv2 ViT-Small mit Patch-Größe 14 |
| $d_\text{model}$ | 384 | Embedding-Dimension |
| $h$ | 6 | Anzahl Attention-Heads |
| $d_k = d_v$ | 64 | Dimension pro Head ($d_\text{model} / h$) |
| $L$ | 12 | Anzahl Transformer-Blöcke |
| $p$ | 14 | Patch-Größe in Pixeln |
| Eingabebild | $224 \times 224 \times 3$ | RGB |
| Encoder-Transform | $196 \times 196$ | Resize vor DINOv2 ($\lfloor 224/16 \rfloor \times 14$) |
| Patch-Raster | $14 \times 14 = 196$ | Anzahl Patches nach Transform |

---

## 1. DINOv2 Attention Map

### Ziel

Visualisierung der Self-Attention-Gewichte des CLS-Tokens auf alle Bild-Patches
in der letzten Transformer-Schicht. Zeigt, welche Bildregionen das Netzwerk als
besonders informativ für die globale Bildrepräsentation erachtet.

### Ablauf

#### Schritt 1: Bildvorbereitung

Das Eingabebild $\mathbf{I} \in \mathbb{R}^{224 \times 224 \times 3}$ wird durch
die `encoder_transform` auf $196 \times 196$ Pixel resized und als Tensor
normalisiert (ImageNet-Mittelwert und -Standardabweichung):

$$\mathbf{I}_\text{norm} = \frac{\mathbf{I} - \boldsymbol{\mu}_\text{ImageNet}}{\boldsymbol{\sigma}_\text{ImageNet}}, \quad \mathbf{I}_\text{norm} \in \mathbb{R}^{1 \times 3 \times 196 \times 196}$$

#### Schritt 2: Patch-Tokenisierung

DINOv2 zerlegt das Bild in nicht-überlappende Patches der Größe $p \times p = 14 \times 14$ Pixel.
Bei einer Bildgröße von $196 \times 196$ ergibt sich ein Raster von:

$$N_h = N_w = \frac{196}{14} = 14 \quad \Rightarrow \quad N = N_h \times N_w = 196 \text{ Patches}$$

Jeder Patch $\mathbf{p}_i \in \mathbb{R}^{14 \times 14 \times 3}$ wird durch eine lineare
Projektion (Convolutional Layer mit Kernel-Größe $14 \times 14$, Stride $14$) in einen
Embedding-Vektor $\mathbf{e}_i \in \mathbb{R}^{384}$ transformiert:

$$\mathbf{e}_i = \mathbf{W}_\text{proj} \cdot \text{flatten}(\mathbf{p}_i) + \mathbf{b}_\text{proj}$$

Zusätzlich wird ein spezieller **CLS-Token** $\mathbf{e}_\text{CLS} \in \mathbb{R}^{384}$
vorangestellt und Positionsembeddings addiert:

$$\mathbf{X}^{(0)} = [\mathbf{e}_\text{CLS};\, \mathbf{e}_1;\, \mathbf{e}_2;\, \ldots;\, \mathbf{e}_{196}] + \mathbf{E}_\text{pos}$$

$$\mathbf{X}^{(0)} \in \mathbb{R}^{197 \times 384}$$

#### Schritt 3: Forward durch Transformer-Blöcke 1–11

Die Token-Sequenz durchläuft die ersten 11 von 12 Transformer-Blöcken.
Jeder Block besteht aus Multi-Head Self-Attention (MHSA) und Feed-Forward Network (FFN)
mit LayerNorm und Residualverbindungen:

$$\mathbf{X}^{(\ell)} = \text{FFN}\!\left(\text{MHSA}\!\left(\text{LN}(\mathbf{X}^{(\ell-1)})\right) + \mathbf{X}^{(\ell-1)}\right) + \text{MHSA}(\ldots) + \mathbf{X}^{(\ell-1)}$$

für $\ell = 1, \ldots, 11$.

#### Schritt 4: Attention-Extraktion im letzten Block ($\ell = 12$)

Im letzten Block wird die Attention explizit berechnet statt nur das Ergebnis weiterzugeben.

**4a) LayerNorm:**

$$\hat{\mathbf{X}} = \text{LayerNorm}(\mathbf{X}^{(11)})$$

**4b) QKV-Projektion:**

Für jeden der 197 Tokens werden **Query**, **Key** und **Value** Vektoren berechnet:

$$[\mathbf{Q};\, \mathbf{K};\, \mathbf{V}] = \hat{\mathbf{X}} \cdot \mathbf{W}_\text{QKV} + \mathbf{b}_\text{QKV}$$

mit $\mathbf{W}_\text{QKV} \in \mathbb{R}^{384 \times 1152}$ (drei mal 384).

**4c) Multi-Head Aufteilung:**

Die projizierte Matrix wird in $h = 6$ Heads aufgeteilt, indem der gesamte
Embedding-Raum in 6 gleich große Unterräume von je $d_k = 64$ Dimensionen zerlegt wird:

$$\mathbf{Q}^{(j)},\, \mathbf{K}^{(j)},\, \mathbf{V}^{(j)} \in \mathbb{R}^{197 \times 64}, \quad j = 0, \ldots, 5$$

Die Dimension wird via Reshape aufgeteilt (kein Kopieren, nur Neuinterpretation):

$$\text{reshape:} \quad (1, 197, 384) \rightarrow (1, 197, 6, 64) \rightarrow (1, 6, 197, 64)$$

**4d) Scaled Dot-Product Attention pro Head:**

Für jeden Head $j$ wird die Attention-Matrix berechnet:

$$\mathbf{A}^{(j)} = \text{softmax}\!\left(\frac{\mathbf{Q}^{(j)} {\mathbf{K}^{(j)}}^\top}{\sqrt{d_k}}\right) \in \mathbb{R}^{197 \times 197}$$

Der Skalierungsfaktor $\frac{1}{\sqrt{d_k}} = \frac{1}{\sqrt{64}} = \frac{1}{8}$ verhindert,
dass die Dot-Products bei hoher Dimensionalität zu groß werden und die Softmax-Funktion
in Sättigungsbereiche treibt.

Jeder Eintrag $A^{(j)}_{i,k}$ beschreibt, wie stark Token $i$ auf Token $k$ "achtet".

#### Schritt 5: CLS-Token Attention extrahieren

Der CLS-Token (Index 0) aggregiert globale Information über das gesamte Bild.
Seine Attention-Gewichte auf die 196 Patch-Tokens zeigen, welche Bildregionen
für die Gesamtrepräsentation am wichtigsten sind:

$$\mathbf{a}_\text{CLS}^{(j)} = \mathbf{A}^{(j)}[0,\, 1\!:\!197] \in \mathbb{R}^{196}$$

für jeden Head $j = 0, \ldots, 5$.

Register-Tokens (falls vorhanden, bei DINOv2 v2) werden dabei übersprungen:
$n_\text{reg} = 0$ bei `dinov2_vits14`.

#### Schritt 6: Reshape zum 2D-Raster

Die 196 Attention-Werte werden in das Patch-Raster zurückgeformt:

$$\mathbf{a}_\text{CLS}^{(j)} \in \mathbb{R}^{196} \xrightarrow{\text{reshape}} \mathbf{M}^{(j)} \in \mathbb{R}^{14 \times 14}$$

#### Schritt 7: Upsampling und Overlay

1. Bilineare Interpolation auf Originalbildgröße: $14 \times 14 \rightarrow 224 \times 224$
2. Normalisierung auf $[0, 1]$
3. Einfärben mit Colormap (`inferno`: schwarz → lila → orange → gelb)
4. Halbtransparentes Overlay ($\alpha = 0.55$) über das Originalbild

#### Ausgabe

- **6 Head-Heatmaps**: Jeder Head zeigt seine individuelle Attention-Verteilung
- **1 Mean-Heatmap**: Durchschnitt aller 6 Heads $\bar{\mathbf{M}} = \frac{1}{6}\sum_{j=0}^{5} \mathbf{M}^{(j)}$
- Obere Reihe: Rohe Attention in Patch-Auflösung
- Untere Reihe: Interpoliertes Overlay auf dem Originalbild

---

## 2. DINOv2 PCA Feature Map

### Ziel

Visualisierung der semantischen Struktur der Patch-Token-Repräsentationen.
Patches mit ähnlichen Features erhalten ähnliche Farben — man sieht dadurch,
welche Bildregionen das Netzwerk als zusammengehörig versteht.

### Ablauf

#### Schritt 1: Patch-Token-Extraktion

Die 196 Patch-Tokens werden aus dem vollständigen Forward-Pass des DINOv2 extrahiert
(nach LayerNorm, Feature-Key `x_norm_patchtokens`):

$$\mathbf{F} = [\mathbf{f}_1;\, \mathbf{f}_2;\, \ldots;\, \mathbf{f}_{196}] \in \mathbb{R}^{196 \times 384}$$

Jeder Vektor $\mathbf{f}_i \in \mathbb{R}^{384}$ ist die semantische Repräsentation
des $i$-ten Patches.

#### Schritt 2: PCA-Dimensionsreduktion

Principal Component Analysis reduziert die 384 Dimensionen auf die 3 Hauptkomponenten
mit der höchsten Varianz:

$$\text{PCA}: \mathbb{R}^{196 \times 384} \rightarrow \mathbb{R}^{196 \times 3}$$

Die drei Hauptkomponenten $\text{PC}_1, \text{PC}_2, \text{PC}_3$ sind die Eigenvektoren der
Kovarianzmatrix $\mathbf{C} = \frac{1}{N}\sum_{i=1}^{N}(\mathbf{f}_i - \bar{\mathbf{f}})(\mathbf{f}_i - \bar{\mathbf{f}})^\top$,
sortiert nach absteigenden Eigenwerten $\lambda_1 \geq \lambda_2 \geq \lambda_3$.

Die erklärte Varianz jeder Komponente ist $\frac{\lambda_k}{\sum_i \lambda_i}$.

#### Schritt 3: Normalisierung und Farbzuweisung

Jede PCA-Komponente wird auf $[0, 1]$ normalisiert und als RGB-Farbkanal interpretiert:

$$\text{R}_i = \text{norm}(\text{PC}_1(\mathbf{f}_i)), \quad
\text{G}_i = \text{norm}(\text{PC}_2(\mathbf{f}_i)), \quad
\text{B}_i = \text{norm}(\text{PC}_3(\mathbf{f}_i))$$

mit $\text{norm}(x) = \frac{x - x_\min}{x_\max - x_\min}$.

#### Schritt 4: Reshape und Upsampling

$$\mathbf{P} \in \mathbb{R}^{196 \times 3} \xrightarrow{\text{reshape}} \mathbb{R}^{14 \times 14 \times 3} \xrightarrow{\text{bilinear}} \mathbb{R}^{224 \times 224 \times 3}$$

#### Ausgabe

- **Links**: Original-Bild
- **Mitte**: PCA Feature Map in Patch-Auflösung ($14 \times 14$)
- **Rechts**: PCA Feature Map bilinear interpoliert auf $224 \times 224$
- **Titel**: Erklärte Varianz pro Hauptkomponente

#### Interpretation

Gleichfarbige Regionen = semantisch ähnliche Feature-Repräsentationen.
Typisch: Würfel, Tisch und Hintergrund erscheinen in klar unterschiedlichen Farben.

---

## 3. DINOv2 Feature Similarity Map

### Ziel

Für ausgewählte Referenz-Patches die Cosinus-Ähnlichkeit zu allen 196 Patches darstellen.
Zeigt, welche Bildregionen dasselbe Objekt oder Material repräsentieren.

### Ablauf

#### Schritt 1: Patch-Tokens extrahieren und normalisieren

$$\hat{\mathbf{f}}_i = \frac{\mathbf{f}_i}{\|\mathbf{f}_i\|_2}, \quad i = 1, \ldots, 196$$

#### Schritt 2: Referenz-Patches auswählen

Es werden 4 automatische Referenz-Positionen im Patch-Raster gewählt:

| Referenz | Position $(r, c)$ | Typische Bildregion |
|----------|-------------------|---------------------|
| R0 | $(7, 7)$ — Mitte | Würfel / Roboter |
| R1 | $(3, 3)$ — Oben-links | Hintergrund |
| R2 | $(10, 7)$ — Unten-Mitte | Tisch |
| R3 | $(3, 10)$ — Oben-rechts | Hintergrund |

#### Schritt 3: Cosinus-Ähnlichkeit berechnen

Für jede Referenz $\mathbf{r}$ wird die Ähnlichkeit zu allen Patches berechnet:

$$\text{sim}(\mathbf{r}, \mathbf{f}_i) = \frac{\mathbf{r} \cdot \mathbf{f}_i}{\|\mathbf{r}\| \cdot \|\mathbf{f}_i\|} = \hat{\mathbf{r}} \cdot \hat{\mathbf{f}}_i \in [-1, 1]$$

In Matrixform: $\mathbf{s} = \hat{\mathbf{F}} \cdot \hat{\mathbf{r}}^\top \in \mathbb{R}^{196}$

#### Schritt 4: Reshape und Visualisierung

$$\mathbf{s} \in \mathbb{R}^{196} \xrightarrow{\text{reshape}} \mathbb{R}^{14 \times 14} \xrightarrow{\text{bilinear}} \mathbb{R}^{224 \times 224}$$

Colormap `RdYlBu_r`: Rot = hohe Ähnlichkeit ($\approx 1$), Blau = niedrige Ähnlichkeit ($\approx 0$).

#### Ausgabe

- **Original** mit farbig markierten Referenzpunkten
- **Pro Referenz**: Similarity-Heatmap in Patch-Auflösung + Overlay auf Originalbild
- 2 Reihen × (1 + Anzahl Referenzen) Spalten

---

## 4. ViT Predictor Attention

### Ziel

Visualisierung der Self-Attention im trainierten ViT-Predictor — dem Teil des World Models,
der den nächsten Frame vorhersagt. Zeigt, welche räumlichen Regionen für die
Zustandsvorhersage besonders relevant sind.

### Unterschied zum DINOv2

- DINOv2: **Frozen**, vortrainierter Encoder — generische visuelle Features
- ViT Predictor: **Trainiert** auf die Franka-Cube-Stacking-Aufgabe — aufgabenspezifische Attention

### Ablauf

#### Schritt 1: Kontext-Frames vorbereiten

$T_\text{hist} = 4$ aufeinanderfolgende Frames werden als Kontext bereitgestellt.
Jeder Frame wird durch den DINOv2 Encoder (mit `encoder_transform`) zu 196 Patch-Tokens
encodiert. Die Actions zwischen den Frames werden durch den Action-Encoder verarbeitet,
die EEF-Positionen (Proprio) durch den Proprio-Encoder.

$$\mathbf{Z} = [\underbrace{\mathbf{z}_1^{(1)}, \ldots, \mathbf{z}_{196}^{(1)}}_{\text{Frame 1}},\, \ldots,\, \underbrace{\mathbf{z}_1^{(4)}, \ldots, \mathbf{z}_{196}^{(4)}}_{\text{Frame 4}}]$$

Bei `concat_dim=1` (Konfiguration) werden Action- und Proprio-Embeddings entlang
der Feature-Dimension auf die Patch-Tokens addiert/konkateniert, sodass die Sequenzlänge
$4 \times 196 = 784$ Tokens beträgt.

#### Schritt 2: Kausale Self-Attention

Der ViT Predictor verwendet **kausale Attention**: Jeder Frame kann nur auf vorherige
Frames zugreifen (Autoregressive Maske). Die Attention im letzten Layer wird über
Forward-Hooks extrahiert:

$$\mathbf{A}_\text{pred}^{(j)} = \text{softmax}\!\left(\frac{\mathbf{Q}^{(j)} {\mathbf{K}^{(j)}}^\top}{\sqrt{d_k}} + \mathbf{M}_\text{kausal}\right)$$

wobei $\mathbf{M}_\text{kausal}$ eine globale Maske ist, die Future-Tokens auf $-\infty$ setzt.

#### Schritt 3: Self-Attention des letzten Frames extrahieren

Aus der Attention-Matrix wird der Block des letzten Kontext-Frames isoliert:

$$\mathbf{A}_\text{last} = \mathbf{A}_\text{pred}[t_4\!:\!t_4\!+\!196,\; t_4\!:\!t_4\!+\!196] \in \mathbb{R}^{196 \times 196}$$

mit $t_4 = 3 \times 196 = 588$.

Die Summe über die Key-Dimension zeigt, welche Patches insgesamt am meisten
Attention erhalten (besonders informativ für die Vorhersage):

$$\mathbf{s}_i = \sum_{k=1}^{196} A_\text{last}[k, i] \quad \Rightarrow \quad \mathbf{S} \in \mathbb{R}^{14 \times 14}$$

#### Schritt 4: Reshape und Overlay

Identisch zu DINOv2: $14 \times 14 \rightarrow 224 \times 224$ via bilineare Interpolation,
Colormap `inferno`, $\alpha = 0.55$ Overlay.

#### Ausgabe

- **Original Frame** + **Self-Attention Summe** (Mean über alle Heads)
- **Pro Head** ($\leq 4$): Individuelle Attention-Heatmap + Overlay
- **Attention Matrix**: Vollständige $(784 \times 784)$ Matrix als Heatmap (Mean über Heads)

---

## 5. VQ-VAE Rekonstruktion

### Ziel

Vergleich des Original-Zielbilds mit der Ausgabe des VQ-VAE Decoders.
Zeigt die interne Repräsentationsqualität des World Models.

### Ablauf

#### Schritt 1: Encoding

Die Kontext-Frames und Actions werden durch den Encoder verarbeitet:

$$\mathbf{Z} = \text{VWorldModel.encode}(\text{obs}, \text{act}) \in \mathbb{R}^{B \times T \times N \times d}$$

#### Schritt 2: Decoding

Der VQ-VAE Decoder rekonstruiert RGB-Bilder aus den latenten Repräsentationen:

$$\hat{\mathbf{I}}_t = \text{VQ-VAE}_\text{dec}(\mathbf{Z}[:, t, :, :]) \in \mathbb{R}^{3 \times H \times W}$$

Der Decoder verwendet Vector Quantization: Die kontinuierlichen Latent-Vektoren werden
auf einen diskreten Codebook abgebildet, und aus den quantisierten Vektoren wird
das Bild rekonstruiert.

#### Schritt 3: Denormalisierung und Vergleich

Die Rekonstruktion und das Original werden von der ImageNet-Normalisierung zurücktransformiert:

$$\mathbf{I}_\text{out} = \mathbf{I}_\text{norm} \cdot \boldsymbol{\sigma}_\text{ImageNet} + \boldsymbol{\mu}_\text{ImageNet}$$

und als uint8-Bilder dargestellt.

#### Schritt 4: MSE-Berechnung

Pro Frame wird der Mean Squared Error berechnet:

$$\text{MSE}_t = \frac{1}{3HW} \sum_{c,h,w} \left(\mathbf{I}_t[c,h,w] - \hat{\mathbf{I}}_t[c,h,w]\right)^2$$

#### Ausgabe

- **Obere Reihe**: Original-Frames $t = 0, \ldots, T-1$
- **Untere Reihe**: VQ-VAE-Rekonstruktionen mit MSE-Wert pro Frame

---

## Multi-Head Self-Attention — Detaillierte Erklärung

### Was ist Self-Attention?

Self-Attention ist der Mechanismus, mit dem jedes Token in einer Sequenz lernt,
welche anderen Tokens für seine eigene Repräsentation relevant sind. Im Kontext
von Vision Transformers: Jeder Bild-Patch "entscheidet", auf welche anderen Patches
er achten soll.

### Warum Multi-Head?

Statt eine einzelne Attention über den vollen $d_\text{model} = 384$-dimensionalen
Raum zu berechnen, wird dieser in $h = 6$ unabhängige Unterräume ("Heads")
von je $d_k = 64$ Dimensionen aufgeteilt:

$$\underbrace{384}_{\text{Gesamt}} = \underbrace{6}_{\text{Heads}} \times \underbrace{64}_{\text{pro Head}}$$

**Motivation**: Jeder Head kann sich auf einen anderen Aspekt der visuellen Information
spezialisieren. Verschiedene Heads lernen typischerweise:

- **Lokale Muster**: Kanten, Texturen, Formen in der direkten Nachbarschaft
- **Farbinformation**: Farbkontraste zwischen Regionen (z.B. roter Würfel vs.~grauer Tisch)
- **Semantische Objekte**: Zusammengehörende Regionen desselben Objekts
- **Globale Struktur**: Räumliche Beziehungen zwischen weit entfernten Patches

### Formale Definition

Für Head $j \in \{0, \ldots, 5\}$:

**1. Lineare Projektion** in den Unterraum:

$$\mathbf{Q}^{(j)} = \mathbf{X} \mathbf{W}_Q^{(j)}, \quad \mathbf{K}^{(j)} = \mathbf{X} \mathbf{W}_K^{(j)}, \quad \mathbf{V}^{(j)} = \mathbf{X} \mathbf{W}_V^{(j)}$$

mit $\mathbf{W}_Q^{(j)}, \mathbf{W}_K^{(j)}, \mathbf{W}_V^{(j)} \in \mathbb{R}^{384 \times 64}$

(In der Implementierung wird eine einzelne Matrix $\mathbf{W}_\text{QKV} \in \mathbb{R}^{384 \times 1152}$
verwendet und das Ergebnis anschließend aufgeteilt.)

**2. Attention-Gewichte** berechnen:

$$\mathbf{A}^{(j)} = \text{softmax}\!\left(\frac{\mathbf{Q}^{(j)} {\mathbf{K}^{(j)}}^\top}{\sqrt{64}}\right) \in \mathbb{R}^{197 \times 197}$$

- $\mathbf{Q}^{(j)} {\mathbf{K}^{(j)}}^\top$: Dot-Product misst die Ähnlichkeit zwischen Queries und Keys
- $\frac{1}{\sqrt{64}}$: Skalierung verhindert Sättigung der Softmax
- Softmax: Normalisierung zu einer Wahrscheinlichkeitsverteilung ($\sum_k A_{i,k} = 1$)

**3. Gewichtete Aggregation**:

$$\text{head}_j = \mathbf{A}^{(j)} \mathbf{V}^{(j)} \in \mathbb{R}^{197 \times 64}$$

Jedes Token wird als gewichtete Summe aller Value-Vektoren neu berechnet,
wobei die Gewichte die Attention-Scores sind.

**4. Konkatenation und Projektion**:

$$\text{MHSA}(\mathbf{X}) = [\text{head}_0;\, \text{head}_1;\, \ldots;\, \text{head}_5] \cdot \mathbf{W}_O$$

mit $\mathbf{W}_O \in \mathbb{R}^{384 \times 384}$.

### Q, K, V — Intuition

| Vektor | Rolle | Analogie |
|--------|-------|----------|
| **Query** ($\mathbf{Q}$) | "Wonach suche ich?" | Eine Frage, die jedes Token stellt |
| **Key** ($\mathbf{K}$) | "Was biete ich an?" | Ein Label, das jedes Token trägt |
| **Value** ($\mathbf{V}$) | "Welche Information trage ich?" | Der tatsächliche Inhalt |

Der Dot-Product $\mathbf{q}_i \cdot \mathbf{k}_j$ misst, wie gut die "Frage" von Token $i$
zum "Angebot" von Token $j$ passt. Hoher Wert → Token $j$ ist relevant für Token $i$
→ sein Value $\mathbf{v}_j$ fließt stärker in die neue Repräsentation von Token $i$ ein.

### CLS-Token — Rolle und Bedeutung

Der CLS-Token ist ein spezieller, **lernbarer** Token, der an Position 0 der Sequenz steht.
Er hat kein zugeordnetes Bild-Patch, sondern dient als **globaler Aggregator**:

- Er stellt Queries an alle 196 Patch-Tokens
- Seine Attention-Gewichte $\mathbf{a}_\text{CLS}^{(j)} \in \mathbb{R}^{196}$ zeigen,
  welche Bildregionen er als besonders informativ erachtet
- Seine finale Repräsentation $\mathbf{f}_\text{CLS} \in \mathbb{R}^{384}$ ist
  die komprimierte Zusammenfassung des gesamten Bildes

Durch die Visualisierung der CLS-Attention sehen wir, **wohin das Modell schaut**,
um das Bild zu verstehen.

---

## Zusammenfassung der Dimensionen

```
Eingabe:         (1, 3, 224, 224)         RGB-Bild
                        ↓
encoder_transform:  (1, 3, 196, 196)      Resize auf Modell-Eingabegröße
                        ↓
Patch Embedding: (1, 197, 384)            196 Patches + 1 CLS, je 384-dim
                        ↓
12× Transformer: (1, 197, 384)            Self-Attention + FFN pro Block
                        ↓
CLS-Attention:   (6, 196)                 6 Heads × 196 Patch-Gewichte
                        ↓
Reshape:         (6, 14, 14)              Raster-Form
                        ↓
Upsampling:      (6, 224, 224)            Bilineare Interpolation
                        ↓
Overlay:         (6, 224, 224, 4)         RGBA-Heatmap auf Originalbild
```

---

## Nutzung

```bash
conda activate dino_wm
cd ~/Desktop/dino_wm

# Mit trainiertem Modell (alle 6 Visualisierungen)
python visualize_features.py --model_name 260305/07-56 --episode_idx 0 --frame_idx 5

# Nur DINOv2 (ohne trainiertes Modell)
python visualize_features.py --image_path /pfad/zu/bild.png
```

Ausgabe in: `feature_visualizations/<model_name>/`
