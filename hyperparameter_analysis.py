#!/usr/bin/env python3
"""
DINO World Model – Hyperparameter-Analyse & Optimierung
=========================================================

Dieses Script analysiert und optimiert die Hyperparameter für das DINO WM Training
auf einer NVIDIA A5000 (24 564 MiB VRAM).

Variablen:
  - batch_size       (effektive Batch-Größe pro GPU)
  - num_workers      (DataLoader Worker – nur Ladegeschwindigkeit)
  - T                (Episodenlänge / Anzahl Timesteps)
  - num_episodes     (Anzahl Episoden im Datensatz)
  - frameskip        (temporales Subsampling)
  - num_hist         (Anzahl Kontext-Frames)
  - VRAM             (GPU-Speicher der A5000 = 24 564 MiB)

Prioritäten:
  1. num_hist maximieren (bessere temporale Kontextinformation)
  2. Effizienz maximieren (weniger Train-Samples = schnellere Epochen)

Harte Grenze:
  - VRAM ≤ 24 564 MiB

Outputs:
  - Parameter-Sweep-Tabellen (CSV + Konsole)
  - Grafiken für Masterarbeit (PDF + PNG)
  - Optimale Konfiguration pro Szenario

Autor: Jacob Weyer
Datum: 15.02.2026
"""

import os
import math
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless-Backend für Server
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

# ============================================================================
#  KONSTANTEN: DINO WM Architektur (aus Code + Paper)
# ============================================================================

# --- GPU ---
VRAM_TOTAL_MIB = 24_564          # A5000 VRAM
VRAM_SAFETY_MARGIN = 0.90        # 90% nutzbar (OS, CUDA Overhead, Fragmentation)
VRAM_USABLE_MIB = int(VRAM_TOTAL_MIB * VRAM_SAFETY_MARGIN)
VRAM_USABLE_BYTES = VRAM_USABLE_MIB * 1024 * 1024

# --- Validierungs-Overhead ---
# val() in train.py hat 3 VRAM-Quellen die sich stapeln:
#   1. Feste Kosten (Model/Optimizer/Gradients) bleiben nach Training im VRAM
#   2. openloop_rollout: torch.no_grad(), aber rollout() wächst z-Tensor via torch.cat
#      + decode_obs() erzeugt Bilder → Speicherfragmentierung im CUDA-Allocator
#   3. Validation Loop: model(obs, act) OHNE torch.no_grad()!
#      → Voller Computation Graph wird gebaut (Autograd Overhead) obwohl backward() nie folgt
#      → Identische Activation-Kosten wie Training Forward Pass
#   4. Kein torch.cuda.empty_cache() zwischen Rollout und Val-Loop
#      → CUDA-Allocator-Fragmentierung: freie Blöcke nicht zusammenhängend
#
# Konsequenz: Die Validierung braucht MEHR VRAM als das Training!
#   Training: Forward + Backward → Activations werden nach Backward freigegeben
#   Validation: Rollout-Residuen + Forward (mit Graph!) → Peak-Belastung
VAL_ROLLOUT_NUM = 10             # openloop_rollout: 10 Rollouts pro Dataset
VAL_ROLLOUT_CONTEXTS = 2         # pro Rollout: num_hist + 1-frame Start
VAL_FRAGMENTATION_FACTOR = 1.12  # ~12% CUDA-Allocator-Fragmentierung (empirisch)

# --- DINOv2 Encoder (frozen, ViT-S/14) ---
DINO_PARAMS = 21_000_000         # 21M Parameter (ViT-S/14 distilled)
DINO_EMB_DIM = 384               # Embedding-Dimension
DINO_PATCH_SIZE = 14             # Patch-Größe
IMG_SIZE = 224                   # Input-Bildgröße
# Encoder gibt 196 Patches (14×14) aus, danach durch VQVAE decoder_scale=16
# → num_side_patches = 224 // 16 = 14, num_patches = 14² = 196
NUM_PATCHES = (IMG_SIZE // 16) ** 2  # = 196

# --- ViT Predictor ---
VIT_DEPTH = 6                    # Transformer Depth
VIT_HEADS = 16                   # Attention Heads
VIT_MLP_DIM = 2048               # MLP Dimension
VIT_DIM_HEAD = 64                # Dimension per Head

# --- VQ-VAE Decoder ---
VQVAE_CHANNEL = 384              # Decoder channel
VQVAE_N_RES_BLOCK = 4            # ResBlocks
VQVAE_N_RES_CHANNEL = 128        # ResBlock channels

# --- Action/Proprio Encoder ---
ACTION_DIM = 6                   # [x_start, y_start, z_start, x_end, y_end, z_end]
PROPRIO_DIM = 3                  # [x, y, z] EE position
ACTION_EMB_DIM = 10
PROPRIO_EMB_DIM = 10
NUM_ACTION_REPEAT = 1
NUM_PROPRIO_REPEAT = 1
CONCAT_DIM = 1                   # concat along feature dim

# --- Training ---
NUM_PRED = 1                     # Immer 1 (Code-Constraint)
GRADIENT_ACCUMULATION = 1        # Standard
SPLIT_RATIO = 0.9                # Train/Val Split

# --- Paper-Referenz (Zhou et al. 2025) ---
PAPER_BATCH_SIZE = 32
PAPER_EPOCHS = 100
PAPER_NUM_HIST_RANGE = [1, 3]    # H=1 oder H=3 je nach Env
PAPER_FRAMESKIP_RANGE = [1, 5]


# ============================================================================
#  VRAM-SCHÄTZUNG
# ============================================================================

def estimate_model_params() -> Dict[str, int]:
    """Schätzt die Parameteranzahl jeder Modell-Komponente."""
    
    # DINOv2 Encoder (frozen, aber im VRAM)
    encoder_params = DINO_PARAMS
    
    # ViT Predictor
    # concat_dim=1: dim = 384 + 10*1 + 10*1 = 404 → patches = 196
    # concat_dim=0: dim = 384, patches = 196 + 2 = 198
    if CONCAT_DIM == 1:
        predictor_dim = DINO_EMB_DIM + PROPRIO_EMB_DIM * NUM_PROPRIO_REPEAT + ACTION_EMB_DIM * NUM_ACTION_REPEAT
        predictor_patches = NUM_PATCHES
    else:
        predictor_dim = DINO_EMB_DIM
        predictor_patches = NUM_PATCHES + 2
    
    # ViT Parameters:
    # pos_embedding: num_frames * num_patches * dim
    # per layer: 2 * (LayerNorm + Attention + FeedForward)
    # Attention: to_qkv(dim → inner_dim*3) + to_out(inner_dim → dim) + norm
    # FeedForward: norm + linear(dim → mlp_dim) + linear(mlp_dim → dim)
    inner_dim = VIT_DIM_HEAD * VIT_HEADS  # 64 * 16 = 1024
    attn_params = predictor_dim + predictor_dim * inner_dim * 3 + inner_dim * predictor_dim + predictor_dim  # QKV + out + norms
    ff_params = predictor_dim * 2 + predictor_dim * VIT_MLP_DIM + VIT_MLP_DIM * predictor_dim  # norms + linear
    per_layer = attn_params + ff_params
    predictor_params = per_layer * VIT_DEPTH + predictor_dim  # + final norm
    
    # VQVAE Decoder
    # upsample_b: Decoder(384, 384, 384, 4, 128, stride=4)
    # dec: Decoder(384, 3, 384, 4, 128, stride=4)
    # Each has: Conv2d(in, channel, 3) + 4*ResBlock(channel, 128) + ConvT stride=4
    # ResBlock: Conv2d(channel, 128, 3) + Conv2d(128, channel, 1) ≈ channel*128*9 + 128*channel
    resblock_params = VQVAE_CHANNEL * VQVAE_N_RES_CHANNEL * 9 + VQVAE_N_RES_CHANNEL * VQVAE_CHANNEL
    decoder_block_params = (
        VQVAE_CHANNEL * VQVAE_CHANNEL * 9 +           # input conv
        VQVAE_N_RES_BLOCK * resblock_params +          # resblocks
        VQVAE_CHANNEL * (VQVAE_CHANNEL // 2) * 16 +   # ConvT stride 2
        (VQVAE_CHANNEL // 2) * VQVAE_CHANNEL * 16     # ConvT stride 2 (zweite Stufe)
    )
    decoder_params = decoder_block_params * 2  # upsample_b + dec
    
    # Action Encoder (ProprioceptiveEmbedding mit Conv1d)
    action_encoder_params = ACTION_DIM * ACTION_EMB_DIM + ACTION_EMB_DIM
    proprio_encoder_params = PROPRIO_DIM * PROPRIO_EMB_DIM + PROPRIO_EMB_DIM
    
    # Quantize (frozen wenn quantize=False)
    quantize_params = DINO_EMB_DIM * 2048  # embed buffer
    
    return {
        'encoder (frozen)': encoder_params,
        'predictor': predictor_params,
        'decoder': decoder_params,
        'action_encoder': action_encoder_params,
        'proprio_encoder': proprio_encoder_params,
        'quantize (buffer)': quantize_params,
    }


def estimate_vram_mib(batch_size: int, num_hist: int, frameskip: int) -> Dict[str, float]:
    """
    Schätzt den VRAM-Verbrauch in MiB für eine gegebene Konfiguration.
    
    Methode: Empirisch kalibriertes analytisches Modell.
    
    Kalibrierungsdaten (gemessen auf A5000, 24564 MiB):
    ┌───────────┬────────┬────────┬──────────────────┬─────────┐
    │ batch_size │ num_hist│frameskip│ VRAM (MiB)       │ Quelle  │
    ├───────────┼────────┼────────┼──────────────────┼─────────┤
    │     4     │   6    │   2    │ 16467 (gemessen) │ Epoch 1 │
    │     8     │   3-4  │   2    │ ~14738 (~60%)    │ train.yaml│
    │    16     │   3-4  │   2    │ >24564 (OOM!)    │ train.yaml│
    │    32     │   3-4  │   2    │ >24564 (OOM!)    │ train.yaml│
    └───────────┴────────┴────────┴──────────────────┴─────────┘
    
    Das Modell zerlegt den VRAM in:
      1. Feste Kosten: Model Weights + Optimizer + Gradients (~559 MiB)
      2. Lineare Kosten: Encoder + Decoder Activations ∝ batch_size × num_frames
      3. Quadratische Kosten: ViT Predictor Attention ∝ batch_size × (num_hist × 196)²
      4. Konstante Kosten: CUDA Context + Fragmentation (~2000 MiB)
    
    Referenzen:
    - EleutherAI: Transformer Math 101 (https://blog.eleuther.ai/transformer-math/)
    - HuggingFace: GPU Training Best Practices
    - NVIDIA: Performance Tuning Guide für Tensor Cores
    """
    params = estimate_model_params()
    total_trainable = params['predictor'] + params['decoder'] + params['action_encoder'] + params['proprio_encoder']
    total_frozen = params['encoder (frozen)'] + params['quantize (buffer)']
    
    num_frames = num_hist + NUM_PRED  # Gesamtzahl der Frames pro Sample
    
    # ═══════════════════════════════════════════════════════════════════
    # 1. FESTE KOSTEN: Model Weights + Optimizer + Gradients
    # ═══════════════════════════════════════════════════════════════════
    # Frozen: fp32 (4 bytes) — kein Optimizer, keine Gradients
    # Trainable: fp16 forward (2 bytes) + AdamW states (12 bytes) + Gradients (2 bytes)
    model_bytes = total_frozen * 4 + total_trainable * 2
    optimizer_bytes = total_trainable * 12  # AdamW: fp32 master + momentum + variance
    gradient_bytes = total_trainable * 2    # fp16 gradients
    
    # ═══════════════════════════════════════════════════════════════════
    # 2. AKTIVIERUNGEN (batch-abhängig, HAUPTTREIBER)
    # ═══════════════════════════════════════════════════════════════════
    # 
    # Empirische Kalibrierung:
    #   Gemessen: bs=4, nh=6, fs=2 → 16467 MiB total
    #   Feste Kosten: ~559 MiB (Model+Optimizer+Gradients)
    #   CUDA Overhead: ~2000 MiB
    #   → Activations: 16467 - 559 - 2000 = ~13908 MiB
    #
    # Die Activations setzen sich zusammen aus:
    #   A) Encoder (DINOv2): linear in batch_size × num_frames
    #   B) Predictor (ViT):  quadratisch in seq_len = num_hist × 196
    #   C) Decoder (VQVAE):  linear in batch_size × num_frames
    #   D) Misc:             tiling, rearrange, loss buffers
    #
    # PyTorch Overhead-Faktoren (nicht in theoretischen Formeln):
    #   - Autograd Graph: speichert Computation Graph für Backward (~2×)
    #   - CUDA Allocator Fragmentation: Speicherfragmentierung (~30%)
    #   - cuDNN Workspace: für optimierte Conv-Kernel
    #   - einops rearrange: temporäre Tensoren bei reshape/tiling
    #   - torch.cat: Allokation neuer Tensoren
    
    # --- 2a. DINOv2 Encoder ---
    # Forward Pass: (batch × num_frames) Bilder durch ViT-S/14
    # Trotz frozen: Computation Graph wird für Backward (Decoder/Predictor) gespeichert
    # ViT-S/14: 12 Layers, 6 Heads, dim=384, MLP=1536, patches=196
    dino_heads = 6
    dino_layers = 12
    # Input tensors
    enc_input = batch_size * num_frames * 3 * IMG_SIZE * IMG_SIZE * 2
    # Per-layer activations: QKV, attention scores, FF intermediates
    enc_attn = batch_size * num_frames * dino_heads * NUM_PATCHES * NUM_PATCHES * 2 * dino_layers
    enc_ff = batch_size * num_frames * NUM_PATCHES * 1536 * 2 * dino_layers
    # Output embeddings
    enc_output = batch_size * num_frames * NUM_PATCHES * DINO_EMB_DIM * 2
    encoder_activation = enc_input + enc_attn + enc_ff + enc_output
    
    # --- 2b. ViT Predictor ---
    if CONCAT_DIM == 1:
        pred_dim = DINO_EMB_DIM + PROPRIO_EMB_DIM * NUM_PROPRIO_REPEAT + ACTION_EMB_DIM * NUM_ACTION_REPEAT
        pred_patches = NUM_PATCHES
    else:
        pred_dim = DINO_EMB_DIM
        pred_patches = NUM_PATCHES + 2
    
    seq_len = num_hist * pred_patches  # num_hist × 196 = LANG!
    inner_dim = VIT_DIM_HEAD * VIT_HEADS  # 64 × 16 = 1024
    
    # QKV + Output projections (per layer)
    pred_qkv = batch_size * seq_len * inner_dim * 3 * 2 * VIT_DEPTH
    # Attention matrix: O(n²) — DER HAUPTTREIBER BEI HOHEM num_hist
    attention_mem = batch_size * VIT_HEADS * seq_len * seq_len * 2 * VIT_DEPTH
    # MLP intermediates
    pred_mlp = batch_size * seq_len * VIT_MLP_DIM * 2 * VIT_DEPTH
    # Residuals + LayerNorm saved states
    pred_residual = batch_size * seq_len * pred_dim * 2 * VIT_DEPTH * 3
    predictor_activation = pred_qkv + attention_mem + pred_mlp + pred_residual
    
    # --- 2c. VQVAE Decoder ---
    # 2× Forward Pass: prediction decode + reconstruction decode
    dec_passes = num_frames * 2
    # Pipeline: 14×14 → upsample_b(stride=4) → 56×56 → dec(stride=4) → 224×224
    # Each spatial resolution stores activations for backward
    dec_14 = batch_size * dec_passes * VQVAE_CHANNEL * 14 * 14 * 2
    dec_resblocks_14 = batch_size * dec_passes * VQVAE_N_RES_CHANNEL * 14 * 14 * 2 * VQVAE_N_RES_BLOCK * 2
    dec_28 = batch_size * dec_passes * (VQVAE_CHANNEL // 2) * 28 * 28 * 2
    dec_56 = batch_size * dec_passes * VQVAE_CHANNEL * 56 * 56 * 2
    dec_resblocks_56 = batch_size * dec_passes * VQVAE_N_RES_CHANNEL * 56 * 56 * 2 * VQVAE_N_RES_BLOCK * 2
    dec_112 = batch_size * dec_passes * (VQVAE_CHANNEL // 2) * 112 * 112 * 2
    dec_224 = batch_size * dec_passes * 3 * IMG_SIZE * IMG_SIZE * 2
    decoder_activation = dec_14 + dec_resblocks_14 + dec_28 + dec_56 + dec_resblocks_56 + dec_112 + dec_224
    
    # --- 2d. Misc: Loss, Tiling, einops ---
    loss_mem = batch_size * num_frames * 3 * IMG_SIZE * IMG_SIZE * 4  # fp32 MSE
    tiling_mem = batch_size * num_frames * NUM_PATCHES * pred_dim * 2 * 3
    misc_activation = loss_mem + tiling_mem
    
    # --- Theoretische Summe ---
    theoretical_activation = encoder_activation + predictor_activation + decoder_activation + misc_activation
    
    # ═══════════════════════════════════════════════════════════════════
    # 3. EMPIRISCHE KALIBRIERUNG
    # ═══════════════════════════════════════════════════════════════════
    # Theoretische Formeln unterschätzen systematisch, weil:
    #   - PyTorch Autograd Graph Overhead (Backward speichert Computation Graph)
    #   - CUDA Memory Allocator Fragmentation (cudaMalloc Granularität)
    #   - Temporäre Tensoren bei einops.rearrange, torch.cat, repeat
    #   - cuDNN Workspace für optimierte Convolution-Kernel
    #   - Gradient Accumulation Buffers
    #
    # Kalibrierung an Messpunkt: bs=4, nh=6, fs=2 → 16467 MiB
    #   Feste Kosten (model+opt+grad): 559 MiB
    #   CUDA Overhead: 2000 MiB
    #   → Echte Activations: 16467 - 559 - 2000 = 13908 MiB
    #   → Theoretische Activations bei diesem Punkt: X MiB
    #   → Kalibrierungsfaktor = 13908 / X
    
    # Berechne den Kalibrierungsfaktor am Referenzpunkt
    # Referenzpunkt: bs=4, nh=6, fs=2
    ref_bs, ref_nh, ref_fs = 4, 6, 2
    ref_num_frames = ref_nh + NUM_PRED  # 7
    ref_seq_len = ref_nh * pred_patches  # 6 × 196 = 1176
    
    # Referenz-Activations (gleiche Formeln wie oben)
    ref_enc = (ref_bs * ref_num_frames * 3 * IMG_SIZE * IMG_SIZE * 2 +
               ref_bs * ref_num_frames * dino_heads * NUM_PATCHES * NUM_PATCHES * 2 * dino_layers +
               ref_bs * ref_num_frames * NUM_PATCHES * 1536 * 2 * dino_layers +
               ref_bs * ref_num_frames * NUM_PATCHES * DINO_EMB_DIM * 2)
    ref_pred = (ref_bs * ref_seq_len * inner_dim * 3 * 2 * VIT_DEPTH +
                ref_bs * VIT_HEADS * ref_seq_len * ref_seq_len * 2 * VIT_DEPTH +
                ref_bs * ref_seq_len * VIT_MLP_DIM * 2 * VIT_DEPTH +
                ref_bs * ref_seq_len * pred_dim * 2 * VIT_DEPTH * 3)
    ref_dec_passes = ref_num_frames * 2
    ref_dec = (ref_bs * ref_dec_passes * VQVAE_CHANNEL * 14 * 14 * 2 +
               ref_bs * ref_dec_passes * VQVAE_N_RES_CHANNEL * 14 * 14 * 2 * VQVAE_N_RES_BLOCK * 2 +
               ref_bs * ref_dec_passes * (VQVAE_CHANNEL // 2) * 28 * 28 * 2 +
               ref_bs * ref_dec_passes * VQVAE_CHANNEL * 56 * 56 * 2 +
               ref_bs * ref_dec_passes * VQVAE_N_RES_CHANNEL * 56 * 56 * 2 * VQVAE_N_RES_BLOCK * 2 +
               ref_bs * ref_dec_passes * (VQVAE_CHANNEL // 2) * 112 * 112 * 2 +
               ref_bs * ref_dec_passes * 3 * IMG_SIZE * IMG_SIZE * 2)
    ref_misc = (ref_bs * ref_num_frames * 3 * IMG_SIZE * IMG_SIZE * 4 +
                ref_bs * ref_num_frames * NUM_PATCHES * pred_dim * 2 * 3)
    ref_theoretical = ref_enc + ref_pred + ref_dec + ref_misc
    
    # Gemessene Activations am Referenzpunkt
    ref_fixed_costs_mib = 559   # Model + Optimizer + Gradients
    ref_cuda_overhead_mib = 2000
    ref_measured_total_mib = 16467
    ref_measured_activations_mib = ref_measured_total_mib - ref_fixed_costs_mib - ref_cuda_overhead_mib
    ref_measured_activations_bytes = ref_measured_activations_mib * 1024 * 1024
    
    # Kalibrierungsfaktor
    calibration_factor = ref_measured_activations_bytes / ref_theoretical
    
    # Kalibrierte Activations
    calibrated_activation = theoretical_activation * calibration_factor
    
    # ═══════════════════════════════════════════════════════════════════
    # 4. CUDA OVERHEAD (Basis-Kosten, unabhängig von Konfiguration)
    # ═══════════════════════════════════════════════════════════════════
    # PyTorch CUDA Context + cuDNN + Allocator Reservierung
    cuda_overhead = 2000 * 1024 * 1024
    
    # ═══════════════════════════════════════════════════════════════════
    # 5. GESAMT
    # ═══════════════════════════════════════════════════════════════════
    total_bytes = model_bytes + optimizer_bytes + gradient_bytes + calibrated_activation + cuda_overhead
    
    return {
        'model_weights_mib': model_bytes / (1024**2),
        'optimizer_mib': optimizer_bytes / (1024**2),
        'gradients_mib': gradient_bytes / (1024**2),
        'activations_mib': calibrated_activation / (1024**2),
        'cuda_overhead_mib': cuda_overhead / (1024**2),
        'total_mib': total_bytes / (1024**2),
        'total_trainable_params': total_trainable,
        'total_frozen_params': total_frozen,
        'predictor_seq_len': seq_len,
        'attention_matrix_mib': (attention_mem * calibration_factor) / (1024**2),
        'vram_utilization_pct': (total_bytes / (1024**2)) / VRAM_TOTAL_MIB * 100,
        'calibration_factor': round(calibration_factor, 2),
    }


def estimate_vram_validation_peak_mib(
    batch_size: int, num_hist: int, frameskip: int, 
    T: int = 22, num_episodes: int = 500
) -> Dict[str, float]:
    """
    Schätzt den VRAM-Spitzenverbrauch während der Validierungsphase.
    
    Die Validierung (val() in train.py) ist der kritische VRAM-Engpass:
    
    Ablauf von val():
    ┌──────────────────────────────────────────────────────────────────────┐
    │ 1. model.eval()                                                    │
    │ 2. with torch.no_grad():                                           │
    │    ├─ openloop_rollout(train_traj_dset, 10 rollouts)               │
    │    │  └─ pro Rollout: encode → predict → cat → predict → cat → ... │
    │    │     + decode_obs() → erzeugt Bilder auf GPU                   │
    │    └─ openloop_rollout(val_traj_dset, 10 rollouts)                 │
    │       └─ gleicher Ablauf                                           │
    │                                                                    │
    │ 3. for batch in valid_dataloader:  ← KEIN torch.no_grad()!         │
    │    └─ model(obs, act)              ← baut vollen Computation Graph │
    │       ├─ encode(obs, act)          ← DINOv2 forward + embeddings   │
    │       ├─ predict(z_src)            ← ViT Attention (O(n²))         │
    │       └─ decode(z_pred)            ← VQVAE decode (2× passes)     │
    │    └─ encode_obs(obs) [plot only]  ← ZUSÄTZLICHER Encoder-Pass     │
    └──────────────────────────────────────────────────────────────────────┘
    
    VRAM-Lastspitze entsteht durch:
    A) Feste Kosten:    Model + Optimizer + Gradients (bleiben nach Training)
    B) Rollout-Residuen: CUDA-Allocator hält freigegebene Blöcke fragmentiert
    C) Val Forward Pass: GLEICHE Activations wie Training (weil kein no_grad)!
    D) Extra Decode:     1. Batch decoded für Plot-Erzeugung
    E) Fragmentierung:   ~12% Overhead durch nicht-zusammenhängende Blöcke
    
    Returns:
        Dict mit Training-VRAM, Validierungs-Peak und Overhead-Analyse
    """
    # Basis: Training-VRAM
    train_vram = estimate_vram_mib(batch_size, num_hist, frameskip)
    train_total = train_vram['total_mib']
    train_activations = train_vram['activations_mib']
    
    # ═══════════════════════════════════════════════════════════════════
    # A) Rollout-Overhead (openloop_rollout unter torch.no_grad())
    # ═══════════════════════════════════════════════════════════════════
    # model.rollout() baut z-Tensor iterativ auf:
    #   z = encode(obs_0) → (1, n_past, 196, 404)
    #   loop: z_pred = predict(z[-num_hist:]) → z = cat(z, z_new)
    #   → z wächst auf (1, horizon, 196, 404)
    # 
    # Maximaler Rollout-Horizon: max_horizon = (T - 1) // frameskip
    max_horizon = max(1, (T - 1) // frameskip)
    # z-Tensor am Ende eines Rollouts: (1, horizon, 196, 404) × fp16
    z_rollout_bytes = 1 * max_horizon * NUM_PATCHES * 404 * 2
    # + decodierte Bilder: (1, horizon, 3, 224, 224) × fp32
    vis_rollout_bytes = 1 * max_horizon * 3 * IMG_SIZE * IMG_SIZE * 4
    # + Predictor-Aktivierungen pro Step (unter no_grad → kein Graph, aber Tensorallokation)
    # predict() verarbeitet z[-num_hist:] → seq_len = min(num_hist, current_len) × 196
    # Unter no_grad: nur Forward-Tensoren, kein Backward-Graph → ~40% der Training-Activations
    pred_seq_len = num_hist * (NUM_PATCHES if CONCAT_DIM == 1 else NUM_PATCHES + 2)
    inner_dim = VIT_DIM_HEAD * VIT_HEADS
    rollout_predict_bytes = (
        1 * VIT_HEADS * pred_seq_len * pred_seq_len * 2 * VIT_DEPTH +  # Attention
        1 * pred_seq_len * inner_dim * 3 * 2 * VIT_DEPTH +             # QKV
        1 * pred_seq_len * VIT_MLP_DIM * 2 * VIT_DEPTH                 # MLP
    )
    
    # Pro Rollout: z + visuals + predict (letzteres wird pro Step allokiert/freigegeben)
    # 10 Rollouts × 2 Kontexte (num_hist + 1-frame), aber sequentiell → nicht kumulativ
    # ABER: CUDA-Allocator gibt Blöcke nicht ans OS zurück → Fragmentierung!
    single_rollout_mib = (z_rollout_bytes + vis_rollout_bytes + rollout_predict_bytes) / (1024**2)
    # Worst case: letzter Rollout-Speicher + Fragmentierung aller vorherigen
    rollout_residual_mib = single_rollout_mib * 2  # Faktor 2: Peak bei Decode + nächste Allokation
    
    # ═══════════════════════════════════════════════════════════════════
    # B) Validation Forward Pass (OHNE torch.no_grad())
    # ═══════════════════════════════════════════════════════════════════
    # model(obs, act) baut den VOLLEN Computation Graph!
    # → Identische Activation-Kosten wie Training Forward Pass
    # Der Backward-Graph wird nie genutzt, verschwendet aber VRAM
    # 
    # Training Activations ≈ Val Activations (gleicher batch_size, gleiche Architektur)
    # ABER: Training räumt nach backward() auf → Val räumt NIE auf!
    # → Alle Activations bleiben bis zum nächsten Iteration-Scope liegen
    val_forward_activations_mib = train_activations  # Identisch, weil kein no_grad()
    
    # ═══════════════════════════════════════════════════════════════════
    # C) Extra-Kosten beim ersten Batch (plot=True)
    # ═══════════════════════════════════════════════════════════════════
    # Bei i==0 (plot=True) wird ZUSÄTZLICH aufgerufen:
    #   encode_obs(obs)    → nochmal DINOv2 Forward für z_gt
    #   err_eval()         → z_pred vs z_gt Vergleich
    #   eval_images()      → Bild-Metriken (PSNR, SSIM, etc.)
    # 
    # encode_obs Forward Pass (unter eval, aber OHNE no_grad):
    extra_encode_bytes = batch_size * (num_hist + NUM_PRED) * NUM_PATCHES * DINO_EMB_DIM * 2
    # eval_images: hält obs + visual_out + visual_reconstructed gleichzeitig
    extra_images_bytes = batch_size * (num_hist + NUM_PRED) * 3 * IMG_SIZE * IMG_SIZE * 4 * 3
    extra_plot_mib = (extra_encode_bytes + extra_images_bytes) / (1024**2)
    
    # ═══════════════════════════════════════════════════════════════════
    # D) CUDA Fragmentierung
    # ═══════════════════════════════════════════════════════════════════
    # Nach openloop_rollout: viele kleine Blöcke freigegeben, aber
    # CUDA-Allocator hält sie als fragmentierte Cache-Blöcke
    # Kein torch.cuda.empty_cache() vor dem Val-Loop!
    # → ~12% Overhead auf die Gesamtallokation
    
    # ═══════════════════════════════════════════════════════════════════
    # GESAMT: Validierungs-Peak
    # ═══════════════════════════════════════════════════════════════════
    # Feste Kosten bleiben (Model + Optimizer + Gradients)
    fixed_costs = train_vram['model_weights_mib'] + train_vram['optimizer_mib'] + train_vram['gradients_mib']
    cuda_overhead = train_vram['cuda_overhead_mib']
    
    # Peak = Fixed + CUDA + Val-Activations + Rollout-Residuen + Extra-Plot
    val_peak_before_frag = (
        fixed_costs + cuda_overhead + 
        val_forward_activations_mib + 
        rollout_residual_mib + 
        extra_plot_mib
    )
    
    # Mit Fragmentierung
    val_peak_total = val_peak_before_frag * VAL_FRAGMENTATION_FACTOR
    
    return {
        # Training-Baseline
        'train_total_mib': train_total,
        'train_activations_mib': train_activations,
        'train_vram_pct': train_total / VRAM_TOTAL_MIB * 100,
        # Validierungs-Overhead-Komponenten
        'fixed_costs_mib': fixed_costs,
        'cuda_overhead_mib': cuda_overhead,
        'val_forward_activations_mib': val_forward_activations_mib,
        'rollout_residual_mib': rollout_residual_mib,
        'extra_plot_mib': extra_plot_mib,
        'fragmentation_factor': VAL_FRAGMENTATION_FACTOR,
        # Validierungs-Peak
        'val_peak_mib': val_peak_total,
        'val_peak_pct': val_peak_total / VRAM_TOTAL_MIB * 100,
        'val_overhead_mib': val_peak_total - train_total,
        'val_overhead_pct': (val_peak_total - train_total) / train_total * 100,
        # Sicherheits-Bewertung
        'val_fits_in_vram': val_peak_total <= VRAM_TOTAL_MIB,
        'val_fits_with_margin': val_peak_total <= VRAM_USABLE_MIB,
    }


# ============================================================================
#  TRAINING CONSTRAINTS (aus bisheriger Analyse)
# ============================================================================

def slices_per_episode(T: int, num_hist: int, frameskip: int) -> int:
    """Anzahl Training-Slices pro Episode."""
    needed = (num_hist + NUM_PRED) * frameskip
    return max(0, T - needed + 1)


def training_feasible(T: int, num_hist: int, frameskip: int) -> bool:
    """Kann Training starten? (Slices > 0)"""
    return slices_per_episode(T, num_hist, frameskip) > 0


def rollout_safe(T: int, num_hist: int, frameskip: int) -> bool:
    """Friert openloop_rollout NICHT ein?"""
    min_horizon = 2 + num_hist
    max_horizon = (T - 1) // frameskip
    return max_horizon > min_horizon


def steps_per_epoch(num_episodes: int, T: int, num_hist: int, 
                    frameskip: int, batch_size: int) -> int:
    """Anzahl der Training Steps pro Epoch."""
    train_episodes = int(num_episodes * SPLIT_RATIO)
    slices = slices_per_episode(T, num_hist, frameskip)
    train_samples = train_episodes * slices
    return math.ceil(train_samples / batch_size) if train_samples > 0 else 0


def total_train_samples(num_episodes: int, T: int, num_hist: int, 
                        frameskip: int) -> int:
    """Gesamtanzahl der Training-Samples."""
    train_episodes = int(num_episodes * SPLIT_RATIO)
    slices = slices_per_episode(T, num_hist, frameskip)
    return train_episodes * slices


# ============================================================================
#  PARAMETER-SWEEP
# ============================================================================

@dataclass
class SweepResult:
    """Ergebnis eines Parameter-Sweep-Punktes."""
    batch_size: int
    num_hist: int
    frameskip: int
    T: int
    num_episodes: int
    num_workers: int
    slices: int
    train_samples: int
    steps: int
    training_ok: bool
    rollout_ok: bool
    vram_train_mib: float
    vram_train_pct: float
    vram_val_peak_mib: float
    vram_val_peak_pct: float
    val_overhead_pct: float
    vram_ok: bool
    score: float  # Optimierungs-Score
    attention_seq_len: int
    attention_mib: float


def compute_sweep(
    T_values: List[int] = [20, 21, 22, 25],
    num_hist_values: List[int] = list(range(1, 12)),
    frameskip_values: List[int] = [1, 2, 3, 4, 5],
    batch_size_values: List[int] = [2, 4, 8, 16, 32],
    num_episodes_values: List[int] = [200, 500, 1000],
    num_workers: int = 2,
) -> pd.DataFrame:
    """Führt einen vollständigen Parameter-Sweep durch."""
    
    results = []
    
    for T, num_hist, frameskip, batch_size, num_episodes in itertools.product(
        T_values, num_hist_values, frameskip_values, batch_size_values, num_episodes_values
    ):
        t_ok = training_feasible(T, num_hist, frameskip)
        r_ok = rollout_safe(T, num_hist, frameskip)
        slices = slices_per_episode(T, num_hist, frameskip)
        samples = total_train_samples(num_episodes, T, num_hist, frameskip)
        steps = steps_per_epoch(num_episodes, T, num_hist, frameskip, batch_size)
        
        vram = estimate_vram_mib(batch_size, num_hist, frameskip)
        val_peak = estimate_vram_validation_peak_mib(
            batch_size, num_hist, frameskip, T, num_episodes)
        # Validierungs-Peak ist die harte Grenze!
        vram_ok = val_peak['val_peak_mib'] <= VRAM_TOTAL_MIB
        
        # Score-Berechnung: Priorität 1 = num_hist, Priorität 2 = weniger Samples (Effizienz)
        # Höherer Score = bessere Konfiguration
        if t_ok and r_ok and vram_ok:
            # num_hist hat 10× das Gewicht der Effizienz
            # Effizienz: weniger Steps = schnellere Epochen → invertieren
            efficiency = 1.0 / max(steps, 1)
            score = num_hist * 1000 + efficiency * 100
        else:
            score = -1  # Ungültige Konfiguration
        
        results.append(SweepResult(
            batch_size=batch_size,
            num_hist=num_hist,
            frameskip=frameskip,
            T=T,
            num_episodes=num_episodes,
            num_workers=num_workers,
            slices=slices,
            train_samples=samples,
            steps=steps,
            training_ok=t_ok,
            rollout_ok=r_ok,
            vram_train_mib=round(vram['total_mib'], 1),
            vram_train_pct=round(vram['vram_utilization_pct'], 1),
            vram_val_peak_mib=round(val_peak['val_peak_mib'], 1),
            vram_val_peak_pct=round(val_peak['val_peak_pct'], 1),
            val_overhead_pct=round(val_peak['val_overhead_pct'], 1),
            vram_ok=vram_ok,
            score=round(score, 2),
            attention_seq_len=vram['predictor_seq_len'],
            attention_mib=round(vram['attention_matrix_mib'], 1),
        ))
    
    df = pd.DataFrame([r.__dict__ for r in results])
    return df


# ============================================================================
#  OPTIMIERUNGS-SOLVER
# ============================================================================

def find_optimal_config(
    T: int,
    num_episodes: int,
    max_batch_size: int = 32,
    num_workers: int = 2,
    verbose: bool = True,
) -> Dict:
    """
    Findet die optimale Konfiguration für gegebene Daten-Parameter.
    
    Strategie:
    1. Iteriere num_hist absteigend (maximiere Kontext)
    2. Für jedes num_hist: finde bestes frameskip (klein = feinere Dynamik)
    3. Für jedes (num_hist, frameskip): finde größte batch_size die ins VRAM passt
    
    Rückgabe: Dict mit optimaler Konfiguration und Metriken
    """
    
    best = None
    all_valid = []
    
    for num_hist in range(20, 0, -1):  # Absteigend: maximiere num_hist
        for frameskip in [1, 2, 3, 4, 5]:
            if not training_feasible(T, num_hist, frameskip):
                continue
            if not rollout_safe(T, num_hist, frameskip):
                continue
            
            # Finde größte batch_size die ins VRAM passt
            for batch_size in [32, 16, 8, 4, 2, 1]:
                if batch_size > max_batch_size:
                    continue
                    
                vram = estimate_vram_mib(batch_size, num_hist, frameskip)
                # Prüfe Validierungs-Peak statt nur Training-VRAM!
                val_peak = estimate_vram_validation_peak_mib(
                    batch_size, num_hist, frameskip, T, num_episodes)
                if val_peak['val_peak_mib'] > VRAM_TOTAL_MIB:  # Harte Grenze: Peak < GPU Total
                    continue
                
                slices = slices_per_episode(T, num_hist, frameskip)
                samples = total_train_samples(num_episodes, T, num_hist, frameskip)
                steps = steps_per_epoch(num_episodes, T, num_hist, frameskip, batch_size)
                
                config = {
                    'num_hist': num_hist,
                    'frameskip': frameskip,
                    'batch_size': batch_size,
                    'T': T,
                    'num_episodes': num_episodes,
                    'slices_per_ep': slices,
                    'train_samples': samples,
                    'steps_per_epoch': steps,
                    'vram_train_mib': round(vram['total_mib'], 1),
                    'vram_train_pct': round(vram['vram_utilization_pct'], 1),
                    'vram_val_peak_mib': round(val_peak['val_peak_mib'], 1),
                    'vram_val_peak_pct': round(val_peak['val_peak_pct'], 1),
                    'val_overhead_pct': round(val_peak['val_overhead_pct'], 1),
                    'attention_seq_len': vram['predictor_seq_len'],
                    'attention_mib': round(vram['attention_matrix_mib'], 1),
                    'num_workers': num_workers,
                }
                all_valid.append(config)
                
                if best is None:
                    best = config
                break  # Größte batch_size gefunden → nächstes frameskip
    
    if verbose and best:
        print(f"\n{'='*70}")
        print(f"  OPTIMALE KONFIGURATION für T={T}, {num_episodes} Episoden")
        print(f"{'='*70}")
        for k, v in best.items():
            print(f"  {k:20s}: {v}")
        print(f"{'='*70}")
    
    return {
        'optimal': best,
        'all_valid': all_valid,
        'total_evaluated': len(all_valid),
    }


# ============================================================================
#  VISUALISIERUNGEN FÜR MASTERARBEIT
# ============================================================================

def create_output_dir():
    """Erstellt den Output-Ordner für Grafiken."""
    outdir = os.path.join(os.path.dirname(__file__), 'hyperparameter_analysis')
    os.makedirs(outdir, exist_ok=True)
    return outdir


def plot_1_heatmap_feasibility(T: int = 22, num_episodes: int = 500, save: bool = True):
    """
    Plot 1: Heatmap – Machbarkeitsraum (num_hist × frameskip)
    Zeigt welche Kombinationen (Training OK, Rollout OK, Beides) funktionieren.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    frameskips = list(range(1, 8))
    num_hists = list(range(1, 16))
    
    data = np.zeros((len(num_hists), len(frameskips)))
    
    for i, nh in enumerate(num_hists):
        for j, fs in enumerate(frameskips):
            t_ok = training_feasible(T, nh, fs)
            r_ok = rollout_safe(T, nh, fs)
            if t_ok and r_ok:
                data[i, j] = 3  # Beides OK
            elif t_ok and not r_ok:
                data[i, j] = 2  # Training OK, Rollout Freeze
            elif not t_ok:
                data[i, j] = 1  # Training unmöglich
            else:
                data[i, j] = 0  # Unmöglich
    
    # Eigene Colormap
    colors = ['#d32f2f', '#ff9800', '#4caf50', '#1b5e20']
    cmap = LinearSegmentedColormap.from_list('feasibility', colors, N=4)
    
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=0, vmax=3, origin='lower')
    
    ax.set_xticks(range(len(frameskips)))
    ax.set_xticklabels(frameskips, fontsize=11)
    ax.set_yticks(range(len(num_hists)))
    ax.set_yticklabels(num_hists, fontsize=11)
    ax.set_xlabel('Frameskip', fontsize=13, fontweight='bold')
    ax.set_ylabel('num_hist (H)', fontsize=13, fontweight='bold')
    ax.set_title(f'Machbarkeitsraum: Training & Rollout (T={T})', fontsize=15, fontweight='bold')
    
    # Annotations mit Slices
    for i, nh in enumerate(num_hists):
        for j, fs in enumerate(frameskips):
            sl = slices_per_episode(T, nh, fs)
            t_ok = training_feasible(T, nh, fs)
            r_ok = rollout_safe(T, nh, fs)
            
            if t_ok and r_ok:
                txt = f'{sl}'
                color = 'white'
            elif t_ok:
                txt = f'{sl}\n⚠'
                color = 'black'
            else:
                txt = '✗'
                color = 'white'
            ax.text(j, i, txt, ha='center', va='center', fontsize=9, color=color, fontweight='bold')
    
    # Legende
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1b5e20', label='Training ✓ + Rollout ✓ (Zahl = Slices/Episode)'),
        Patch(facecolor='#ff9800', label='Training ✓ + Rollout FREEZE ⚠'),
        Patch(facecolor='#d32f2f', label='Training unmöglich ✗'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, 
              framealpha=0.9, fancybox=True)
    
    plt.tight_layout()
    if save:
        outdir = create_output_dir()
        for ext in ['pdf', 'png']:
            fig.savefig(os.path.join(outdir, f'01_feasibility_heatmap_T{T}.{ext}'), 
                        dpi=300, bbox_inches='tight')
    plt.close()
    return fig


def plot_2_vram_vs_batch_numhist(T: int = 22, save: bool = True):
    """
    Plot 2: VRAM-Verbrauch vs. batch_size für verschiedene num_hist
    Zeigt die VRAM-Grenze der A5000.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    batch_sizes = [1, 2, 4, 8, 16, 32]
    frameskip = 2  # Fixiert auf empfohlenen Wert
    colors_nh = plt.cm.viridis(np.linspace(0.1, 0.9, 8))
    
    for idx, num_hist in enumerate(range(1, 9)):
        if not rollout_safe(T, num_hist, frameskip):
            continue
        vram_train = []
        vram_val = []
        for bs in batch_sizes:
            vram = estimate_vram_mib(bs, num_hist, frameskip)
            val_peak = estimate_vram_validation_peak_mib(bs, num_hist, frameskip, T)
            vram_train.append(vram['total_mib'])
            vram_val.append(val_peak['val_peak_mib'])
        ax.plot(batch_sizes, vram_train, 'o-', color=colors_nh[idx], 
                label=f'H={num_hist} (Train)', linewidth=2, markersize=6)
        ax.plot(batch_sizes, vram_val, 's--', color=colors_nh[idx], 
                label=f'H={num_hist} (Val Peak)', linewidth=1.5, markersize=5, alpha=0.6)
    
    # VRAM Limit
    ax.axhline(y=VRAM_USABLE_MIB, color='red', linestyle='--', linewidth=2, 
               label=f'VRAM Limit ({VRAM_USABLE_MIB} MiB @ 90%)')
    ax.axhline(y=VRAM_TOTAL_MIB, color='darkred', linestyle=':', linewidth=1.5,
               label=f'VRAM Total ({VRAM_TOTAL_MIB} MiB)')
    
    ax.fill_between([0.5, 35], VRAM_USABLE_MIB, VRAM_TOTAL_MIB + 2000, 
                    alpha=0.15, color='red', label='_nolegend_')
    
    ax.set_xlabel('Batch Size', fontsize=13, fontweight='bold')
    ax.set_ylabel('Geschätzter VRAM (MiB)', fontsize=13, fontweight='bold')
    ax.set_title(f'VRAM-Verbrauch: batch_size × num_hist (frameskip={frameskip}, T={T})', 
                 fontsize=14, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xticks(batch_sizes)
    ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.8, 40)
    
    plt.tight_layout()
    if save:
        outdir = create_output_dir()
        for ext in ['pdf', 'png']:
            fig.savefig(os.path.join(outdir, f'02_vram_vs_batch_numhist_T{T}.{ext}'), 
                        dpi=300, bbox_inches='tight')
    plt.close()
    return fig


def plot_3_samples_efficiency(T: int = 22, num_episodes: int = 500, save: bool = True):
    """
    Plot 3: Train-Samples und Steps/Epoch für verschiedene (num_hist, frameskip)
    Zeigt die Effizienz-Dimension.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    frameskips = [1, 2, 3, 4, 5]
    batch_size = 4
    
    colors_fs = ['#1565c0', '#2e7d32', '#f57f17', '#e64a19', '#6a1b9a']
    
    # Left: Slices per Episode
    for idx, fs in enumerate(frameskips):
        nhs = []
        slices_vals = []
        for nh in range(1, 15):
            if rollout_safe(T, nh, fs):
                nhs.append(nh)
                slices_vals.append(slices_per_episode(T, nh, fs))
        if nhs:
            ax1.plot(nhs, slices_vals, 's-', color=colors_fs[idx], 
                     label=f'frameskip={fs}', linewidth=2, markersize=8)
            # Mark max num_hist
            ax1.plot(nhs[-1], slices_vals[-1], '*', color=colors_fs[idx], 
                     markersize=18, markeredgecolor='black', markeredgewidth=1)
    
    ax1.set_xlabel('num_hist (H)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Slices pro Episode', fontsize=13, fontweight='bold')
    ax1.set_title(f'Trainings-Effizienz: Slices/Episode (T={T})', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(1, 15))
    
    # Right: Steps per Epoch
    for idx, fs in enumerate(frameskips):
        nhs = []
        steps_vals = []
        for nh in range(1, 15):
            if rollout_safe(T, nh, fs):
                nhs.append(nh)
                steps_vals.append(steps_per_epoch(num_episodes, T, nh, fs, batch_size))
        if nhs:
            ax2.plot(nhs, steps_vals, 'o-', color=colors_fs[idx], 
                     label=f'frameskip={fs}', linewidth=2, markersize=6)
    
    ax2.set_xlabel('num_hist (H)', fontsize=13, fontweight='bold')
    ax2.set_ylabel(f'Steps pro Epoch (batch_size={batch_size})', fontsize=13, fontweight='bold')
    ax2.set_title(f'Trainingszeit: Steps/Epoch ({num_episodes} Ep., T={T})', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(1, 15))
    
    plt.tight_layout()
    if save:
        outdir = create_output_dir()
        for ext in ['pdf', 'png']:
            fig.savefig(os.path.join(outdir, f'03_samples_efficiency_T{T}_E{num_episodes}.{ext}'), 
                        dpi=300, bbox_inches='tight')
    plt.close()
    return fig


def plot_4_optimal_frontier(T: int = 22, save: bool = True):
    """
    Plot 4: Pareto-Front – num_hist vs. Samples-Effizienz
    Zeigt die optimalen Trade-offs zwischen temporalem Kontext und Effizienz.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    episode_counts = [200, 500, 1000]
    markers = ['o', 's', 'D']
    sizes = [80, 100, 120]
    
    for ep_idx, num_ep in enumerate(episode_counts):
        result = find_optimal_config(T, num_ep, verbose=False)
        valid = result['all_valid']
        
        # Gruppiere nach num_hist, nimm jeweils die beste (kleinste steps)
        by_nh = {}
        for cfg in valid:
            nh = cfg['num_hist']
            if nh not in by_nh or cfg['steps_per_epoch'] < by_nh[nh]['steps_per_epoch']:
                by_nh[nh] = cfg
        
        nhs = sorted(by_nh.keys())
        steps = [by_nh[nh]['steps_per_epoch'] for nh in nhs]
        vrams = [by_nh[nh]['vram_val_peak_pct'] for nh in nhs]
        
        scatter = ax.scatter(nhs, steps, c=vrams, cmap='RdYlGn_r', 
                             vmin=20, vmax=100,
                             s=sizes[ep_idx], marker=markers[ep_idx], 
                             edgecolors='black', linewidth=1,
                             label=f'{num_ep} Episoden', zorder=5)
        ax.plot(nhs, steps, '--', alpha=0.4, color='gray', linewidth=1)
        
        # Annotate best
        if by_nh:
            best_nh = max(nhs)
            ax.annotate(f'H={best_nh}\nfs={by_nh[best_nh]["frameskip"]}\nbs={by_nh[best_nh]["batch_size"]}',
                        xy=(best_nh, by_nh[best_nh]['steps_per_epoch']),
                        xytext=(10, 15), textcoords='offset points',
                        fontsize=8, ha='left',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', 
                                  edgecolor='gray', alpha=0.8),
                        arrowprops=dict(arrowstyle='->', color='gray'))
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Val-Peak VRAM (%)', fontsize=11)
    
    ax.set_xlabel('num_hist (H) – Temporaler Kontext', fontsize=13, fontweight='bold')
    ax.set_ylabel('Steps pro Epoch – Trainingszeit ↓ ist besser', fontsize=13, fontweight='bold')
    ax.set_title(f'Optimierungslandschaft: Kontext vs. Effizienz (T={T})', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save:
        outdir = create_output_dir()
        for ext in ['pdf', 'png']:
            fig.savefig(os.path.join(outdir, f'04_optimal_frontier_T{T}.{ext}'), 
                        dpi=300, bbox_inches='tight')
    plt.close()
    return fig


def plot_5_vram_breakdown(T: int = 22, save: bool = True):
    """
    Plot 5: VRAM-Aufschlüsselung als Stacked Bar Chart
    Zeigt woher der VRAM-Verbrauch kommt.
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    configs = [
        (2, 4, 4),   # frameskip=2, num_hist=4, batch_size=4  (konservativ)
        (2, 6, 4),   # frameskip=2, num_hist=6, batch_size=4  (aktuell, gemessen!)
        (2, 7, 4),   # frameskip=2, num_hist=7, batch_size=4  (optimal)
        (2, 7, 8),   # frameskip=2, num_hist=7, batch_size=8  (aggressiv)
        (2, 4, 8),   # frameskip=2, num_hist=4, batch_size=8
        (1, 10, 2),  # frameskip=1, num_hist=10, batch_size=2 (max Kontext)
        (3, 4, 4),   # frameskip=3, num_hist=4, batch_size=4
    ]
    
    labels = [f'fs={fs} H={nh}\nbs={bs}' for fs, nh, bs in configs]
    
    model_vals = []
    optimizer_vals = []
    gradient_vals = []
    activation_vals = []
    overhead_vals = []
    val_overhead_vals = []
    
    for fs, nh, bs in configs:
        vram = estimate_vram_mib(bs, nh, fs)
        val_peak = estimate_vram_validation_peak_mib(bs, nh, fs, T)
        model_vals.append(vram['model_weights_mib'])
        optimizer_vals.append(vram['optimizer_mib'])
        gradient_vals.append(vram['gradients_mib'])
        activation_vals.append(vram['activations_mib'])
        overhead_vals.append(vram['cuda_overhead_mib'])
        val_overhead_vals.append(val_peak['val_overhead_mib'])
    
    x = np.arange(len(configs))
    width = 0.6
    
    bars1 = ax.bar(x, model_vals, width, label='Model Weights', color='#1565c0')
    bars2 = ax.bar(x, optimizer_vals, width, bottom=np.array(model_vals), 
                   label='Optimizer (AdamW)', color='#2e7d32')
    bars3 = ax.bar(x, gradient_vals, width, 
                   bottom=np.array(model_vals) + np.array(optimizer_vals),
                   label='Gradients', color='#f57f17')
    bars4 = ax.bar(x, activation_vals, width, 
                   bottom=np.array(model_vals) + np.array(optimizer_vals) + np.array(gradient_vals),
                   label='Activations (Training)', color='#e64a19')
    bars5 = ax.bar(x, overhead_vals, width, 
                   bottom=np.array(model_vals) + np.array(optimizer_vals) + np.array(gradient_vals) + np.array(activation_vals),
                   label='CUDA Overhead', color='#9e9e9e')
    # Validation Overhead als gestrichelt umrandeter Block oben drauf
    train_tops = np.array(model_vals) + np.array(optimizer_vals) + np.array(gradient_vals) + np.array(activation_vals) + np.array(overhead_vals)
    bars6 = ax.bar(x, val_overhead_vals, width, bottom=train_tops,
                   label='Val-Overhead (Rollout+Fragm.)', color='#b71c1c', alpha=0.35, 
                   edgecolor='#b71c1c', linewidth=1.5, linestyle='--')
    
    # VRAM Limits
    ax.axhline(y=VRAM_USABLE_MIB, color='orange', linestyle='--', linewidth=2, 
               label=f'VRAM 90% ({VRAM_USABLE_MIB} MiB)')
    ax.axhline(y=VRAM_TOTAL_MIB, color='red', linestyle='-', linewidth=2, 
               label=f'VRAM Total ({VRAM_TOTAL_MIB} MiB)')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=0)
    ax.set_ylabel('VRAM (MiB)', fontsize=13, fontweight='bold')
    ax.set_title(f'VRAM-Aufschlüsselung: Training + Validierungs-Peak (T={T})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left', framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Werte auf Bars: Train + Val Peak
    for idx in range(len(configs)):
        train_total = train_tops[idx]
        val_total = train_total + val_overhead_vals[idx]
        color = 'red' if val_total > VRAM_TOTAL_MIB else ('orange' if val_total > VRAM_USABLE_MIB else 'green')
        ax.text(idx, val_total + 200, f'Peak\n{val_total:.0f}', ha='center', va='bottom', 
                fontsize=8, fontweight='bold', color=color)
        ax.text(idx, train_total - 400, f'Train\n{train_total:.0f}', ha='center', va='top', 
                fontsize=7, color='white', fontweight='bold')
    
    plt.tight_layout()
    if save:
        outdir = create_output_dir()
        for ext in ['pdf', 'png']:
            fig.savefig(os.path.join(outdir, f'05_vram_breakdown_T{T}.{ext}'), 
                        dpi=300, bbox_inches='tight')
    plt.close()
    return fig


def plot_6_attention_scaling(save: bool = True):
    """
    Plot 6: Attention Memory Scaling mit num_hist
    Zeigt den quadratischen Anstieg der Attention-Matrix.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Sequence Length vs num_hist
    frameskips = [1, 2, 3, 4]
    colors_fs = ['#1565c0', '#2e7d32', '#f57f17', '#e64a19']
    
    for idx, fs in enumerate(frameskips):
        nhs = list(range(1, 15))
        if CONCAT_DIM == 1:
            seq_lens = [nh * NUM_PATCHES for nh in nhs]
        else:
            seq_lens = [nh * (NUM_PATCHES + 2) for nh in nhs]
        ax1.plot(nhs, seq_lens, 'o-', color=colors_fs[idx], 
                 label=f'frameskip={fs}', linewidth=2, markersize=6)
    
    ax1.set_xlabel('num_hist (H)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Sequence Length (Tokens)', fontsize=12, fontweight='bold')
    ax1.set_title('ViT Predictor: Sequenzlänge', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(1, 15))
    
    # Right: Attention Memory vs num_hist
    batch_size = 4
    for idx, fs in enumerate(frameskips):
        nhs = list(range(1, 15))
        att_mems = []
        for nh in nhs:
            vram = estimate_vram_mib(batch_size, nh, fs)
            att_mems.append(vram['attention_matrix_mib'])
        ax2.plot(nhs, att_mems, 'o-', color=colors_fs[idx], 
                 label=f'frameskip={fs}', linewidth=2, markersize=6)
    
    ax2.set_xlabel('num_hist (H)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Attention Memory (MiB)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Attention-Matrix Speicher (batch_size={batch_size})', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(1, 15))
    
    plt.tight_layout()
    if save:
        outdir = create_output_dir()
        for ext in ['pdf', 'png']:
            fig.savefig(os.path.join(outdir, f'06_attention_scaling.{ext}'), 
                        dpi=300, bbox_inches='tight')
    plt.close()
    return fig


def plot_7_paper_comparison(save: bool = True):
    """
    Plot 7: Vergleich unserer Konfiguration mit dem Paper (Zhou et al. 2025)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Paper-Konfigurationen
    paper_configs = {
        'PointMaze': {'H': 3, 'fs': 5, 'N': 2000, 'T': 100, 'bs': 32},
        'Push-T':    {'H': 3, 'fs': 5, 'N': 18500, 'T': 200, 'bs': 32},
        'Rope':      {'H': 1, 'fs': 1, 'N': 1000, 'T': 5, 'bs': 32},
        'Wall':      {'H': 1, 'fs': 5, 'N': 1920, 'T': 50, 'bs': 32},
    }
    
    # Unsere Konfigurationen (verschiedene Szenarien)
    our_configs = {
        'Franka\n(500ep, T=22)': {'H': 7, 'fs': 2, 'N': 500, 'T': 22, 'bs': 4},
        'Franka\n(1000ep, T=25)': {'H': 5, 'fs': 3, 'N': 1000, 'T': 25, 'bs': 4},
    }
    
    all_labels = list(paper_configs.keys()) + list(our_configs.keys())
    all_H = [c['H'] for c in paper_configs.values()] + [c['H'] for c in our_configs.values()]
    all_fs = [c['fs'] for c in paper_configs.values()] + [c['fs'] for c in our_configs.values()]
    all_N = [c['N'] for c in paper_configs.values()] + [c['N'] for c in our_configs.values()]
    
    x = np.arange(len(all_labels))
    width = 0.25
    
    bars1 = ax.bar(x - width, all_H, width, label='num_hist (H)', color='#1565c0', edgecolor='black')
    bars2 = ax.bar(x, all_fs, width, label='frameskip', color='#f57f17', edgecolor='black')
    
    ax2 = ax.twinx()
    bars3 = ax2.bar(x + width, all_N, width, label='Dataset Size', color='#4caf50', alpha=0.6, edgecolor='black')
    
    # Markierung Paper vs. Ours
    for i in range(len(paper_configs)):
        ax.axvspan(i - 0.45, i + 0.45, alpha=0.08, color='blue')
    for i in range(len(paper_configs), len(all_labels)):
        ax.axvspan(i - 0.45, i + 0.45, alpha=0.08, color='green')
    
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, fontsize=10)
    ax.set_ylabel('num_hist / frameskip', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Dataset Size (Episoden)', fontsize=12, fontweight='bold')
    ax.set_title('Vergleich: Paper (Zhou et al. 2025) vs. Unsere Konfiguration', 
                 fontsize=14, fontweight='bold')
    
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10, framealpha=0.9)
    
    ax.annotate('Paper (A6000 GPU)', xy=(1.5, max(all_H) + 0.5), fontsize=11, 
                ha='center', color='blue', fontweight='bold')
    ax.annotate('Unsere (A5000 GPU)', xy=(len(paper_configs) + 0.5, max(all_H) + 0.5), 
                fontsize=11, ha='center', color='green', fontweight='bold')
    
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    if save:
        outdir = create_output_dir()
        for ext in ['pdf', 'png']:
            fig.savefig(os.path.join(outdir, f'07_paper_comparison.{ext}'), 
                        dpi=300, bbox_inches='tight')
    plt.close()
    return fig


def plot_8_comprehensive_sweep_table(T: int = 22, num_episodes: int = 500, save: bool = True):
    """
    Plot 8: Umfassende Sweep-Tabelle als formatierte Grafik
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')
    
    # Daten sammeln
    rows = []
    batch_size = 4
    for fs in [1, 2, 3, 4, 5]:
        for nh in range(1, 15):
            t_ok = training_feasible(T, nh, fs)
            r_ok = rollout_safe(T, nh, fs)
            if not t_ok:
                continue
            sl = slices_per_episode(T, nh, fs)
            samps = total_train_samples(num_episodes, T, nh, fs)
            st = steps_per_epoch(num_episodes, T, nh, fs, batch_size)
            vram = estimate_vram_mib(batch_size, nh, fs)
            val_peak = estimate_vram_validation_peak_mib(batch_size, nh, fs, T, num_episodes)
            
            # Status: prüfe Val-Peak statt nur Training-VRAM
            if t_ok and r_ok and val_peak['val_peak_mib'] <= VRAM_TOTAL_MIB:
                status = 'OK'
            elif t_ok and r_ok and vram['total_mib'] <= VRAM_TOTAL_MIB:
                status = 'Val OOM!'  # Training passt, aber Val-Peak OOM
            elif t_ok and not r_ok:
                status = 'Freeze'
            else:
                status = 'OOM'
            
            rows.append([
                fs, nh, sl, samps, st, 
                f"{vram['total_mib']:.0f}",
                f"{val_peak['val_peak_mib']:.0f}",
                f"{val_peak['val_peak_pct']:.0f}%",
                status
            ])
    
    columns = ['fs', 'H', 'Sl/Ep', 'Samples', 
               'Steps', 'Train\nVRAM', 'Val Peak\nVRAM', 'Val\n%', 'Status']
    
    table = ax.table(cellText=rows, colLabels=columns, 
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.4)
    
    # Header styling
    for j in range(len(columns)):
        table[(0, j)].set_facecolor('#1565c0')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Row coloring
    for i, row in enumerate(rows, 1):
        status = row[-1]
        if status == 'OK':
            color = '#e8f5e9'  # grün
        elif status == 'Val OOM!':
            color = '#fce4ec'  # rosa — Training passt, aber Val crasht
        elif status == 'Freeze':
            color = '#fff3e0'  # orange
        else:
            color = '#ffebee'  # rot
        for j in range(len(columns)):
            table[(i, j)].set_facecolor(color)
    
    ax.set_title(f'Parameter-Sweep: T={T}, {num_episodes} Ep., bs={batch_size}\n'
                 f'GPU: A5000 ({VRAM_TOTAL_MIB} MiB) | OK = Alles OK | Val OOM! = Val-Peak sprengt VRAM\n'
                 f'Freeze = Rollout eingefroren | OOM = Training OOM',
                 fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    if save:
        outdir = create_output_dir()
        for ext in ['pdf', 'png']:
            fig.savefig(os.path.join(outdir, f'08_sweep_table_T{T}_E{num_episodes}.{ext}'), 
                        dpi=300, bbox_inches='tight')
    plt.close()
    return fig


def plot_9_validation_peak_analysis(T: int = 22, num_episodes: int = 500, save: bool = True):
    """
    Plot 9: Validierungs-Lastspitze – Training vs. Val-Peak VRAM
    
    Layout: 3 Subplots übereinander
      Oben:   Linienplot Training vs Val Peak über num_hist
      Mitte:  Horizontale Balken – Training vs Val Peak (aktuelle Konfig)
      Unten:  Gestapelter Balken – VRAM-Zerlegung des Val Peaks
    """
    fig = plt.figure(figsize=(10, 13))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 0.6, 0.7], hspace=0.38)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    
    frameskip = 2
    batch_size = 4
    
    # ══════════════════════════════════════════════════════════════════════
    # OBEN: Training vs Val Peak über num_hist
    # ══════════════════════════════════════════════════════════════════════
    num_hists = list(range(1, 12))
    
    train_vrams = []
    val_peaks_list = []
    
    for nh in num_hists:
        vram = estimate_vram_mib(batch_size, nh, frameskip)
        val_peak = estimate_vram_validation_peak_mib(batch_size, nh, frameskip, T, num_episodes)
        train_vrams.append(vram['total_mib'])
        val_peaks_list.append(val_peak['val_peak_mib'])
    
    # Fläche zwischen Train und Val Peak
    ax1.fill_between(num_hists, train_vrams, val_peaks_list, alpha=0.20, color='#c62828',
                     label='Val-Overhead')
    ax1.plot(num_hists, train_vrams, 'o-', color='#1565c0', linewidth=2.5, 
             markersize=7, label='Training VRAM', zorder=5)
    ax1.plot(num_hists, val_peaks_list, 's--', color='#c62828', linewidth=2.5, 
             markersize=7, label='Val Peak', zorder=5)
    
    # Limit-Linien
    ax1.axhline(y=VRAM_TOTAL_MIB, color='red', linestyle='-', linewidth=2, 
                label=f'A5000 VRAM ({VRAM_TOTAL_MIB:,} MiB)')
    ax1.axhline(y=VRAM_USABLE_MIB, color='#ef6c00', linestyle='--', linewidth=1.5,
                label=f'90%-Limit ({VRAM_USABLE_MIB:,} MiB)')
    
    # Max machbares num_hist markieren
    max_nh_train = max([nh for nh, v in zip(num_hists, train_vrams) if v <= VRAM_USABLE_MIB], default=0)
    max_nh_val = max([nh for nh, v in zip(num_hists, val_peaks_list) if v <= VRAM_TOTAL_MIB], default=0)
    
    if max_nh_val > 0:
        ax1.axvline(x=max_nh_val, color='#c62828', linestyle=':', linewidth=1.5, alpha=0.6)
        ax1.annotate(f'Max H={max_nh_val}\n(Val-sicher)', 
                     xy=(max_nh_val, val_peaks_list[max_nh_val - 1]),
                     xytext=(-70, 35), textcoords='offset points',
                     fontsize=9, fontweight='bold', color='#c62828',
                     arrowprops=dict(arrowstyle='->', color='#c62828', lw=1.5),
                     bbox=dict(boxstyle='round,pad=0.3', fc='#fce4ec', alpha=0.85))
    if max_nh_train > max_nh_val:
        ax1.annotate(f'Max H={max_nh_train}\n(nur Training)', 
                     xy=(max_nh_train, train_vrams[max_nh_train - 1]),
                     xytext=(15, -35), textcoords='offset points',
                     fontsize=9, fontweight='bold', color='#1565c0',
                     arrowprops=dict(arrowstyle='->', color='#1565c0', lw=1.5),
                     bbox=dict(boxstyle='round,pad=0.3', fc='#e3f2fd', alpha=0.85))
    
    ax1.set_xlabel('num_hist (H)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('VRAM (MiB)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Training vs. Validierungs-Peak  (batch_size={batch_size}, frameskip={frameskip}, T={T})',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper left', framealpha=0.9, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(num_hists)
    ax1.set_ylim(0, max(val_peaks_list) * 1.12)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}k'))
    
    # ══════════════════════════════════════════════════════════════════════
    # MITTE: Horizontale Balken – Training vs Val Peak (aktuelle Konfig)
    # ══════════════════════════════════════════════════════════════════════
    val = estimate_vram_validation_peak_mib(batch_size, 6, frameskip, T, num_episodes)
    
    y_pos = [1.0, 0.4]
    bar_h = 0.35
    ax2.barh(y_pos[0], val['train_total_mib'], height=bar_h, 
             color='#1565c0', alpha=0.85, edgecolor='black', linewidth=0.8)
    ax2.barh(y_pos[1], val['val_peak_mib'], height=bar_h, 
             color='#c62828', alpha=0.85, edgecolor='black', linewidth=0.8)
    
    ax2.axvline(x=VRAM_TOTAL_MIB, color='red', linestyle='-', linewidth=2)
    ax2.axvline(x=VRAM_USABLE_MIB, color='#ef6c00', linestyle='--', linewidth=1.5)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(['Training\n(Fwd+Bwd)', 'Validierung\n(Peak)'],
                        fontsize=11, fontweight='bold')
    ax2.set_xlabel('VRAM (MiB)', fontsize=11, fontweight='bold')
    ax2.set_title(f'Aktuelle Konfiguration: bs={batch_size}, H=6, fs={frameskip}   |   '
                  f'Val-Overhead: +{val["val_overhead_mib"]:.0f} MiB (+{val["val_overhead_pct"]:.0f}%)',
                  fontsize=11, fontweight='bold')
    
    # Werte als Text rechts neben den Balken
    ax2.text(val['train_total_mib'] + 200, y_pos[0],
             f'{val["train_total_mib"]:.0f} MiB  ({val["train_vram_pct"]:.0f}%)',
             va='center', fontsize=10, fontweight='bold', color='#1565c0')
    ax2.text(val['val_peak_mib'] + 200, y_pos[1],
             f'{val["val_peak_mib"]:.0f} MiB  ({val["val_peak_pct"]:.0f}%)',
             va='center', fontsize=10, fontweight='bold', color='#c62828')
    
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.set_xlim(0, max(val['val_peak_mib'] * 1.35, VRAM_TOTAL_MIB * 1.15))
    ax2.set_ylim(0, 1.5)
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}k'))
    
    # ══════════════════════════════════════════════════════════════════════
    # UNTEN: Gestapelter Balken – VRAM-Zerlegung des Val Peaks
    # ══════════════════════════════════════════════════════════════════════
    # Komponenten (vor Fragmentierung) + Fragmentierung separat
    frag_factor = val['fragmentation_factor']
    base_total = val['val_peak_mib'] / frag_factor  # Summe vor ×1.12
    frag_mib = val['val_peak_mib'] - base_total
    
    comp_labels = [
        'Feste Kosten\n(Model+Opt+Grad)',
        'CUDA\nOverhead',
        'Forward Activations\n(kein no_grad!)',
        'Rollout-\nResiduen',
        'Plot-Decode\nExtra',
        'CUDA\nFragmentierung',
    ]
    comp_vals = [
        val['fixed_costs_mib'],
        val['cuda_overhead_mib'],
        val['val_forward_activations_mib'],
        val['rollout_residual_mib'],
        val['extra_plot_mib'],
        frag_mib,
    ]
    comp_colors = ['#1565c0', '#78909c', '#e65100', '#c62828', '#7b1fa2', '#f9a825']
    
    left = 0
    for lbl, v, c in zip(comp_labels, comp_vals, comp_colors):
        bar = ax3.barh(0, v, left=left, height=0.5, color=c, alpha=0.85,
                       edgecolor='black', linewidth=0.5)
        # Beschriftung nur wenn Segment breit genug
        if v > base_total * 0.04:
            ax3.text(left + v / 2, 0, f'{v:.0f}',
                     ha='center', va='center', fontsize=8, fontweight='bold', color='white')
        left += v
    
    # Legende für Komponenten
    from matplotlib.patches import Patch
    legend_patches = [Patch(fc=c, ec='black', lw=0.5, alpha=0.85, label=l.replace('\n', ' '))
                      for l, c, v in zip(comp_labels, comp_colors, comp_vals)]
    ax3.legend(handles=legend_patches, fontsize=8, loc='upper right',
               framealpha=0.9, ncol=3, bbox_to_anchor=(1.0, 1.45))
    
    ax3.axvline(x=VRAM_TOTAL_MIB, color='red', linestyle='-', linewidth=2)
    ax3.axvline(x=VRAM_USABLE_MIB, color='#ef6c00', linestyle='--', linewidth=1.5)
    
    ax3.set_yticks([0])
    ax3.set_yticklabels(['Val Peak\nZerlegung'], fontsize=11, fontweight='bold')
    ax3.set_xlabel('VRAM (MiB)', fontsize=11, fontweight='bold')
    ax3.set_title('VRAM-Zerlegung des Validierungs-Peaks', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.set_xlim(0, max(val['val_peak_mib'] * 1.15, VRAM_TOTAL_MIB * 1.05))
    ax3.set_ylim(-0.5, 0.8)
    ax3.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}k'))
    
    # Ursachen-Box im unteren Plot
    desc_lines = [
        'Ursachen des Val-Overheads:',
        '1. model() OHNE torch.no_grad()',
        '   \u2192 Computation Graph wird gebaut',
        '2. openloop_rollout() vorher:',
        '   \u2192 CUDA-Speicher fragmentiert',
        '3. Kein empty_cache() dazwischen',
        '   \u2192 Bl\u00f6cke nicht zusammenh\u00e4ngend',
    ]
    desc = '\n'.join(desc_lines)
    ax3.text(0.01, -0.42, desc, transform=ax3.transAxes, fontsize=7.5,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.4', fc='#fff9c4', alpha=0.9, ec='#f57f17'))
    
    if save:
        outdir = create_output_dir()
        for ext in ['pdf', 'png']:
            fig.savefig(os.path.join(outdir, f'09_validation_peak_T{T}.{ext}'), 
                        dpi=300, bbox_inches='tight')
    plt.close()
    return fig


def plot_10_episode_length_sweep(save: bool = True):
    """
    Plot 10: Einfluss der Episodenlänge T (10–50) auf Machbarkeit und Effizienz.

    Layout: 2×2 Grid
      Oben-links:  Max num_hist pro T für verschiedene frameskip-Werte
                   (Training-Grenze, Rollout-Grenze, VRAM-Grenze)
      Oben-rechts: Slices pro Episode über T für verschiedene (H, fs) Kombinationen
      Unten-links: Gesamte Train-Samples über T (bei festen Episoden)
      Unten-rechts: "Safe Zone" Heatmap — T × frameskip → max num_hist (Val-sicher)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    ax_tl, ax_tr, ax_bl, ax_br = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    T_range = list(range(10, 51))
    batch_size = 4
    num_episodes = 500

    # ══════════════════════════════════════════════════════════════════════
    # OBEN LINKS: Max num_hist pro T für verschiedene frameskip
    # ══════════════════════════════════════════════════════════════════════
    fs_colors = {1: '#1565c0', 2: '#e65100', 3: '#2e7d32', 4: '#6a1b9a', 5: '#c62828'}

    for fs in [1, 2, 3, 4, 5]:
        max_nh_train = []
        max_nh_rollout = []
        max_nh_vram = []

        for T in T_range:
            # Training-Grenze: num_hist ≤ floor(T/fs) - 1
            nh_train = max(0, T // fs - NUM_PRED)
            # Rollout-Grenze: num_hist < floor((T-1)/fs) - 2
            nh_rollout = max(0, (T - 1) // fs - 2 - 1)  # -1 weil strikt <
            # VRAM-Grenze: höchstes nh wo Val Peak ≤ VRAM_TOTAL
            nh_vram = 0
            for nh_test in range(min(nh_train, nh_rollout), 0, -1):
                vp = estimate_vram_validation_peak_mib(batch_size, nh_test, fs, T, num_episodes)
                if vp['val_peak_mib'] <= VRAM_TOTAL_MIB:
                    nh_vram = nh_test
                    break

            max_nh_train.append(nh_train)
            max_nh_rollout.append(nh_rollout)
            max_nh_vram.append(nh_vram)

        # Effektives Maximum = min(Rollout, VRAM)
        max_nh_effective = [min(r, v) if v > 0 else 0 for r, v in zip(max_nh_rollout, max_nh_vram)]

        ax_tl.plot(T_range, max_nh_effective, '-', color=fs_colors[fs], linewidth=2,
                   label=f'fs={fs}', alpha=0.9)
        # Rollout-Grenze als gepunktete Linie (falls anders als VRAM)
        if max_nh_effective != max_nh_rollout:
            ax_tl.plot(T_range, max_nh_rollout, ':', color=fs_colors[fs], linewidth=1, alpha=0.4)

    # Markiere unsere Konfigurationen
    for T_mark, fs_mark, label in [(22, 2, 'T=22\nfs=2'), (25, 2, 'T=25\nfs=2')]:
        nh_r = max(0, (T_mark - 1) // fs_mark - 2 - 1)
        nh_v = 0
        for nh_test in range(nh_r, 0, -1):
            vp = estimate_vram_validation_peak_mib(batch_size, nh_test, fs_mark, T_mark, num_episodes)
            if vp['val_peak_mib'] <= VRAM_TOTAL_MIB:
                nh_v = nh_test
                break
        eff = min(nh_r, nh_v)
        ax_tl.plot(T_mark, eff, '*', color=fs_colors[fs_mark], markersize=14, zorder=10,
                   markeredgecolor='black', markeredgewidth=0.8)
        ax_tl.annotate(label, xy=(T_mark, eff), xytext=(5, 8),
                       textcoords='offset points', fontsize=8, fontweight='bold',
                       color=fs_colors[fs_mark])

    ax_tl.set_xlabel('Episodenlänge T (Timesteps)', fontsize=11, fontweight='bold')
    ax_tl.set_ylabel('Max. num_hist (H)', fontsize=11, fontweight='bold')
    ax_tl.set_title('Maximales num_hist pro Episodenlänge\n(Rollout + VRAM Limit, bs=4)',
                    fontsize=12, fontweight='bold')
    ax_tl.legend(fontsize=9, title='frameskip', title_fontsize=9, loc='upper left')
    ax_tl.grid(True, alpha=0.3)
    ax_tl.set_xlim(10, 50)
    ax_tl.set_ylim(0, None)

    # ══════════════════════════════════════════════════════════════════════
    # OBEN RECHTS: Slices pro Episode über T
    # ══════════════════════════════════════════════════════════════════════
    configs = [
        (6, 2, '#e65100', 'H=6, fs=2 (aktuell)'),
        (4, 2, '#1565c0', 'H=4, fs=2'),
        (3, 3, '#2e7d32', 'H=3, fs=3'),
        (8, 1, '#6a1b9a', 'H=8, fs=1'),
        (2, 5, '#c62828', 'H=2, fs=5 (Paper-nah)'),
    ]

    for nh, fs, col, lbl in configs:
        slices_list = [slices_per_episode(T, nh, fs) for T in T_range]
        ax_tr.plot(T_range, slices_list, 'o-', color=col, linewidth=2, markersize=3,
                   label=lbl, alpha=0.85)

    ax_tr.axhline(y=1, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Minimum (1 Slice)')
    ax_tr.set_xlabel('Episodenlänge T (Timesteps)', fontsize=11, fontweight='bold')
    ax_tr.set_ylabel('Slices pro Episode', fontsize=11, fontweight='bold')
    ax_tr.set_title('Training-Slices pro Episode über T\n(mehr Slices = mehr Daten pro Episode)',
                    fontsize=12, fontweight='bold')
    ax_tr.legend(fontsize=8, loc='upper left')
    ax_tr.grid(True, alpha=0.3)
    ax_tr.set_xlim(10, 50)
    ax_tr.set_ylim(0, None)

    # ══════════════════════════════════════════════════════════════════════
    # UNTEN LINKS: Gesamte Train-Samples über T
    # ══════════════════════════════════════════════════════════════════════
    for nh, fs, col, lbl in configs:
        samples_list = [total_train_samples(num_episodes, T, nh, fs) for T in T_range]
        ax_bl.plot(T_range, samples_list, 'o-', color=col, linewidth=2, markersize=3,
                   label=lbl, alpha=0.85)

    ax_bl.set_xlabel('Episodenlänge T (Timesteps)', fontsize=11, fontweight='bold')
    ax_bl.set_ylabel('Train-Samples gesamt', fontsize=11, fontweight='bold')
    ax_bl.set_title(f'Gesamte Train-Samples über T\n({num_episodes} Episoden, split_ratio={SPLIT_RATIO})',
                    fontsize=12, fontweight='bold')
    ax_bl.legend(fontsize=8, loc='upper left')
    ax_bl.grid(True, alpha=0.3)
    ax_bl.set_xlim(10, 50)
    ax_bl.set_ylim(0, None)
    ax_bl.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}k' if x >= 1000 else f'{x:.0f}'))

    # ══════════════════════════════════════════════════════════════════════
    # UNTEN RECHTS: Heatmap — T × frameskip → max num_hist (Val-sicher)
    # ══════════════════════════════════════════════════════════════════════
    fs_values = [1, 2, 3, 4, 5]
    T_subset = list(range(10, 51, 2))  # Jeder 2. Wert für Lesbarkeit
    heatmap_data = np.zeros((len(fs_values), len(T_subset)))

    for i, fs in enumerate(fs_values):
        for j, T in enumerate(T_subset):
            nh_rollout = max(0, (T - 1) // fs - 2 - 1)
            nh_vram = 0
            for nh_test in range(min(20, nh_rollout), 0, -1):
                vp = estimate_vram_validation_peak_mib(batch_size, nh_test, fs, T, num_episodes)
                if vp['val_peak_mib'] <= VRAM_TOTAL_MIB:
                    nh_vram = nh_test
                    break
            heatmap_data[i, j] = min(nh_rollout, nh_vram) if nh_vram > 0 else 0

    # Custom Colormap: 0=rot, niedrig=gelb, hoch=grün
    cmap = LinearSegmentedColormap.from_list('rg',
        ['#f44336', '#ff9800', '#ffeb3b', '#8bc34a', '#4caf50', '#1b5e20'], N=256)

    im = ax_br.imshow(heatmap_data, aspect='auto', cmap=cmap, origin='lower',
                      vmin=0, vmax=max(12, heatmap_data.max()))
    ax_br.set_xticks(range(len(T_subset)))
    ax_br.set_xticklabels(T_subset, fontsize=7, rotation=45)
    ax_br.set_yticks(range(len(fs_values)))
    ax_br.set_yticklabels(fs_values)
    ax_br.set_xlabel('Episodenlänge T', fontsize=11, fontweight='bold')
    ax_br.set_ylabel('frameskip', fontsize=11, fontweight='bold')
    ax_br.set_title('Max. num_hist (Val-sicher)\nT × frameskip → H (bs=4)',
                    fontsize=12, fontweight='bold')

    # Werte in Zellen
    for i in range(len(fs_values)):
        for j in range(len(T_subset)):
            val = int(heatmap_data[i, j])
            color = 'white' if val <= 2 else 'black'
            ax_br.text(j, i, str(val), ha='center', va='center',
                       fontsize=7, fontweight='bold', color=color)

    # Markiere unsere Konfigurationen
    for T_mark, fs_mark in [(22, 2), (25, 2)]:
        if T_mark in T_subset:
            j = T_subset.index(T_mark)
            i = fs_values.index(fs_mark)
            ax_br.plot(j, i, 's', markersize=18, markerfacecolor='none',
                       markeredgecolor='white', markeredgewidth=2.5)

    cbar = plt.colorbar(im, ax=ax_br, shrink=0.8)
    cbar.set_label('Max. num_hist', fontsize=10, fontweight='bold')

    fig.suptitle('Einfluss der Episodenlänge T auf die Hyperparameter-Wahl\n'
                 f'GPU: A5000 (24 564 MiB)  |  batch_size={batch_size}  |  {num_episodes} Episoden',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save:
        outdir = create_output_dir()
        for ext in ['pdf', 'png']:
            fig.savefig(os.path.join(outdir, f'10_episode_length_sweep.{ext}'),
                        dpi=300, bbox_inches='tight')
    plt.close()
    return fig


# ============================================================================
#  HAUPTPROGRAMM
# ============================================================================

def main():
    """Hauptprogramm: Führt alle Analysen durch und generiert Grafiken."""
    
    print("=" * 80)
    print("  DINO World Model – Hyperparameter-Analyse & Optimierung")
    print("  GPU: NVIDIA A5000 (24 564 MiB VRAM)")
    print("=" * 80)
    
    # ── 1. Modell-Parameter ──────────────────────────────────────────────
    print("\n📊 MODELL-PARAMETER:")
    params = estimate_model_params()
    total = 0
    for name, count in params.items():
        print(f"  {name:25s}: {count:>12,d} ({count/1e6:.1f}M)")
        total += count
    print(f"  {'GESAMT':25s}: {total:>12,d} ({total/1e6:.1f}M)")
    
    # ── 2. VRAM-Schätzung für aktuelle Konfiguration ─────────────────────
    print("\n📊 VRAM-SCHÄTZUNG (aktuelle Konfiguration: frameskip=2, num_hist=6, batch_size=4):")
    vram = estimate_vram_mib(batch_size=4, num_hist=6, frameskip=2)
    for k, v in vram.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:>10.1f}")
        else:
            print(f"  {k:30s}: {v:>10,d}")
    print(f"\n  Kalibrierung: Faktor {vram['calibration_factor']}× (empirisch, Referenz: 16467 MiB gemessen)")
    print(f"  Cross-Validierung: bs=8,nh=3→59.4% (train.yaml: ~60%) ✅")
    
    # ── 2b. Validierungs-Lastspitze ──────────────────────────────────────
    print("\n📊 VALIDIERUNGS-LASTSPITZE (der kritische Engpass!):")
    val_peak = estimate_vram_validation_peak_mib(4, 6, 2, T=22, num_episodes=500)
    print(f"  {'Training VRAM':30s}: {val_peak['train_total_mib']:>10.0f} MiB ({val_peak['train_vram_pct']:.1f}%)")
    print(f"  {'+ Rollout-Residuen':30s}: {val_peak['rollout_residual_mib']:>10.0f} MiB")
    print(f"  {'+ Val Forward (kein no_grad!)':30s}: {val_peak['val_forward_activations_mib']:>10.0f} MiB")
    print(f"  {'+ Extra Plot-Decode':30s}: {val_peak['extra_plot_mib']:>10.0f} MiB")
    print(f"  {'× Fragmentierung':30s}: {val_peak['fragmentation_factor']:>10.2f}×")
    print(f"  {'─'*52}")
    print(f"  {'VALIDIERUNGS-PEAK':30s}: {val_peak['val_peak_mib']:>10.0f} MiB ({val_peak['val_peak_pct']:.1f}%)")
    print(f"  {'Val-Overhead':30s}: +{val_peak['val_overhead_mib']:>9.0f} MiB (+{val_peak['val_overhead_pct']:.1f}%)")
    print(f"  {'Passt in VRAM?':30s}: {'JA ✅' if val_peak['val_fits_in_vram'] else 'NEIN ❌ OOM!'}")
    print(f"\n  ⚠️  Ursache: val() ruft model(obs,act) OHNE torch.no_grad() auf!")
    print(f"      → Voller Computation Graph wird gebaut, verschwendet VRAM")
    print(f"      → Kein torch.cuda.empty_cache() nach openloop_rollout")
    
    # ── 3. Optimale Konfigurationen ──────────────────────────────────────
    print("\n" + "=" * 80)
    print("  OPTIMIERUNGS-SOLVER: Beste Konfigurationen")
    print("=" * 80)
    
    scenarios = [
        (22, 200, "500ep-Datensatz, 200 Episoden (klein)"),
        (22, 500, "500ep-Datensatz, 500 Episoden (aktuell)"),
        (25, 1000, "1000ep-Datensatz, 1000 Episoden (groß)"),
        (22, 1000, "1000ep-Datensatz mit T=22"),
    ]
    
    optimal_results = {}
    for T, N, desc in scenarios:
        print(f"\n{'─'*70}")
        print(f"  Szenario: {desc}")
        result = find_optimal_config(T, N, verbose=True)
        optimal_results[(T, N)] = result
    
    # ── 4. Vergleich mit Paper ───────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  PAPER-VERGLEICH (Zhou et al. 2025)")
    print("=" * 80)
    print(f"\n  Paper-Defaults: batch_size=32, epochs=100")
    print(f"  Paper-GPU: NVIDIA A6000 (48 GB VRAM)")
    print(f"  Unsere GPU: NVIDIA A5000 (24 GB VRAM) → 50% des VRAM-Budgets")
    print(f"  → batch_size muss reduziert werden")
    print(f"  → Kompensation durch gradient_accumulation möglich")
    print(f"  → Paper H=1-3, Wir: H=5-7 (MEHR Kontext trotz kleinerem VRAM!)")
    
    # ── 5. Parameter-Sweep als CSV ───────────────────────────────────────
    print("\n📊 Generiere Parameter-Sweep...")
    df = compute_sweep(
        T_values=[22, 25],
        num_hist_values=list(range(1, 12)),
        frameskip_values=[1, 2, 3, 4, 5],
        batch_size_values=[4, 8, 16, 32],
        num_episodes_values=[500, 1000],
    )
    
    outdir = create_output_dir()
    csv_path = os.path.join(outdir, 'parameter_sweep.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Sweep gespeichert: {csv_path}")
    print(f"  Gesamt-Konfigurationen: {len(df)}")
    print(f"  Gültige (score > 0): {len(df[df['score'] > 0])}")
    print(f"  Beste score: {df['score'].max():.1f}")
    best_row = df.loc[df['score'].idxmax()]
    print(f"  Beste Konfiguration: num_hist={best_row['num_hist']}, frameskip={best_row['frameskip']}, "
          f"batch_size={best_row['batch_size']}, T={best_row['T']}")
    
    # ── 6. Grafiken generieren ───────────────────────────────────────────
    print("\n📊 Generiere Grafiken für Masterarbeit...")
    
    plots = [
        ("01 Feasibility Heatmap T=22", lambda: plot_1_heatmap_feasibility(T=22)),
        ("01 Feasibility Heatmap T=25", lambda: plot_1_heatmap_feasibility(T=25)),
        ("02 VRAM vs. Batch × num_hist", lambda: plot_2_vram_vs_batch_numhist(T=22)),
        ("03 Samples-Effizienz T=22, 500ep", lambda: plot_3_samples_efficiency(T=22, num_episodes=500)),
        ("03 Samples-Effizienz T=25, 1000ep", lambda: plot_3_samples_efficiency(T=25, num_episodes=1000)),
        ("04 Optimal Frontier T=22", lambda: plot_4_optimal_frontier(T=22)),
        ("04 Optimal Frontier T=25", lambda: plot_4_optimal_frontier(T=25)),
        ("05 VRAM Breakdown T=22", lambda: plot_5_vram_breakdown(T=22)),
        ("06 Attention Scaling", lambda: plot_6_attention_scaling()),
        ("07 Paper-Vergleich", lambda: plot_7_paper_comparison()),
        ("08 Sweep-Tabelle T=22, 500ep", lambda: plot_8_comprehensive_sweep_table(T=22, num_episodes=500)),
        ("08 Sweep-Tabelle T=25, 1000ep", lambda: plot_8_comprehensive_sweep_table(T=25, num_episodes=1000)),
        ("09 Validierungs-Peak T=22", lambda: plot_9_validation_peak_analysis(T=22)),
        ("09 Validierungs-Peak T=25", lambda: plot_9_validation_peak_analysis(T=25)),
        ("10 Episodenlängen-Sweep T=10–50", lambda: plot_10_episode_length_sweep()),
    ]
    
    for name, fn in plots:
        print(f"  ✅ {name}")
        fn()
    
    print(f"\n📁 Alle Grafiken gespeichert in: {outdir}/")
    print(f"   Formate: PDF (Masterarbeit) + PNG (Vorschau)")
    
    # ── 7. Best Practices Zusammenfassung ────────────────────────────────
    print("\n" + "=" * 80)
    print("  BEST PRACTICES & EMPFEHLUNGEN")
    print("=" * 80)
    print("""
  Quellen:
  - Zhou et al. 2025: DINO-WM Paper (Table 11, 12)
  - EleutherAI: Transformer Math 101 
  - HuggingFace: GPU Training Best Practices
  - NVIDIA: Performance Tuning Guide

  1. BATCH SIZE:
     - Paper: 32 (auf A6000 mit 48 GB)
     - Unsere A5000 (24 GB): batch_size=4 ist optimal
     - batch_size sollte Vielfaches von 4 sein (Tensor Core Alignment)
     - HuggingFace empfiehlt: batch_size = 2^n (powers of 2)
     - Größere batch_size → bessere GPU-Auslastung, aber mehr VRAM
     - Kompensation via gradient_accumulation_steps möglich:
       effective_batch = batch_size × gradient_accumulation_steps

  2. NUM_WORKERS:
     - Hat KEINEN Einfluss auf Modell-Qualität oder Steps/Epoch
     - Empfehlung: num_workers = 2-4 (CPU-seitig begrenzt)
     - Zu viele Workers → CPU-Overhead, shared memory Probleme
     - HuggingFace: dataloader_pin_memory=True für schnellere GPU-Transfers

  3. FRAMESKIP:
     - Paper: frameskip=5 für die meisten Envs, frameskip=1 für Rope/Granular
     - Unsere Franka-Daten: frameskip=2 optimal
       → T=22 ist kurz → hoher frameskip verliert zu viel Information
     - Faustrregel: frameskip ≤ T / 10

  4. NUM_HIST:
     - Paper: H=1-3 → wir können H=5-7 nutzen!
     - Mehr Kontext = bessere Vorhersagen (wenn genug Daten)
     - Limitiert durch: VRAM (quadratische Attention) + Rollout-Bug

  5. ATTENTION SCALING:
     - ViT Attention ist O(n²) in der Sequenzlänge
     - seq_len = num_hist × 196 Patches
     - H=7 → 1372 Tokens → Attention Matrix = 7.5M Elemente/Head
     - Das ist der Haupt-VRAM-Treiber!
     
  6. MIXED PRECISION:
     - Accelerator nutzt automatisch Mixed Precision
     - Spart ~40% VRAM bei Activations
     - Empfehlung: bf16 wenn verfügbar, sonst fp16
""")
    
    print("\n✅ Analyse abgeschlossen!")
    print(f"📁 Output-Ordner: {outdir}/")


if __name__ == "__main__":
    main()
