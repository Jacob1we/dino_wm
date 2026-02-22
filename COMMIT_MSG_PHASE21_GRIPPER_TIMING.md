# Commit Message — Phase 21: Gripper-Timing Fix

```
Phase 21: Gripper-Ausführung + temporale Konsistenz (Pick-and-Place)

SYMPTOM:
  Der Roboter erreicht den Würfel jetzt präzise (Phase 20a/b EEF-Fix),
  aber ZWEI Gripper-Probleme verhindern erfolgreiches Greifen:
  1. Der Gripper verpasst den Schließ-Zeitpunkt → Würfel wird nicht gegriffen
  2. Nach erfolgreichem Schließen öffnet sich der Gripper wieder → Würfel fällt

ROOT-CAUSE 1 — CEM ohne temporale Gripper-Konsistenz (cem.py):
  _quantize_gripper() quantisiert zwar korrekt auf {0, 1}, aber behandelt
  JEDEN Horizon-Step UNABHÄNGIG. Eine CEM-Trajektorie kann daher so aussehen:
    Step 1: open → Step 2: open → Step 3: CLOSED → Step 4: OPEN → Step 5: CLOSED
  Das ist physikalisch unsinnig — ein gegriffener Würfel fällt beim Öffnen
  herunter und kann nicht erneut gegriffen werden.
  
  Mathematisch: CEM optimiert rein auf visuelle Feature-Distanz (DINO patches)
  zum Goal-Bild. Die Gripper-Dimension hat keine Semantik im Latent-Space —
  der CEM "versteht" nicht, dass Gripper-Transitionen irreversibel sein müssen.

ROOT-CAUSE 2 — Client ignoriert Gripper-Dimensionen (planning_client.py):
  Der Planning Client extrahiert NUR action[4:7] (EEF-Zielposition) und
  übergibt sie an env.move_ee_to(). Die Gripper-Dimensionen action[3] (g_start)
  und action[7] (g_end) werden KOMPLETT IGNORIERT — env.close_gripper() und
  env.open_gripper() werden NIE aufgerufen.
  
  Konsequenz: Selbst wenn der CEM perfekte Gripper-Aktionen plant, werden
  sie niemals ausgeführt. Der Gripper bleibt immer in seinem Startzustand.

FIX — Drei-Schicht-Architektur:

  Schicht 1: CEM Gripper-Konsistenz (cem.py)
  ─────────────────────────────────────────────
  Neue Methode _enforce_gripper_consistency():
  - Wird NACH _quantize_gripper() aufgerufen (Pipeline: clamp → quantize → consistency)
  - Erzwingt monotone Transition: open* → closed* (kein Zurückspringen)
  - Implementierung via torch.cummax() über die Horizon-Achse:
      is_closed = (action[..., gi] >= mid).float()
      is_closed_sticky, _ = is_closed.cummax(dim=horizon_dim)
    Sobald is_closed einmal True wird, bleibt es für alle folgenden Steps True.
  - Wirkt auf ALLE Gripper-Indices [3, 7, 11, 15] (bei frameskip=2)
  - Physikalische Semantik: "Einmal gegriffen → bleibt gegriffen"
  - Das Öffnen am Zielort ist NICHT Aufgabe des Planners (separater Mechanismus)

  Schicht 2: Gripper-Ausführung (planning_client.py)
  ───────────────────────────────────────────────────
  Neue Funktion execute_gripper_action(gripper_value):
  - Extrahiert action[7] (g_end) aus der 8D Sub-Action
  - Fallback auf action[3] bei kürzeren Action-Vektoren
  - Ruft env.close_gripper() / env.open_gripper() auf
  - env.step(GRIPPER_SETTLE_STEPS=10) nach jeder Zustandsänderung
  - Integriert in BEIDE Modi (Offline + Online/MPC)
  - Gripper State Machine wird pro Episode zurückgesetzt

  Schicht 3: Hysterese gegen Glitches (planning_client.py)
  ─────────────────────────────────────────────────────────
  Problem: Trotz CEM-Konsistenz kann ein einzelner MPC-Re-Plan ein
  fehlerhaftes "open" erzeugen → Würfel fällt während Transport.
  
  Lösung: Gripper State Machine mit Hysterese:
  - Schließen: Sofort beim ersten "close"-Kommando (gripper_value > 0.5)
  - Öffnen: Erst nach HYSTERESIS_THRESHOLD=3 aufeinanderfolgenden "open"-
    Kommandos → einzelne Glitches werden absorbiert
  - Konfigurierbare Parameter:
      GRIPPER_THRESHOLD = 0.5     (close wenn > 0.5)
      HYSTERESIS_THRESHOLD = 3    (3× "open" nötig zum Öffnen)
      GRIPPER_SETTLE_STEPS = 10   (Sim-Steps nach Zustandsänderung)
  - Diagnostik-Output zeigt Gripper-Zustand pro Step: G=C (closed) / G=O (open)

DATENFLUSS (komplett):

  CEM sampelt Action (num_samples, horizon, 16D)
    │
    ├─ _clamp_actions()                  → Workspace-Bounds
    ├─ _quantize_gripper()               → Binär {norm(0), norm(1)}
    ├─ _enforce_gripper_consistency()     → Monoton open* → closed*     ← NEU
    │
    ├─ WM Rollout → Objective Loss → Top-K Selection
    │
    └─ Beste Trajektorie → Server denormalisiert
         │
         └─ Client empfängt 8D Sub-Actions
              │
              ├─ action[4:7] → env.move_ee_to()       (EEF-Position, wie bisher)
              └─ action[7]   → execute_gripper_action() ← NEU
                   │
                   ├─ > 0.5 → env.close_gripper() + settle (sofort)
                   └─ ≤ 0.5 → env.open_gripper() + settle (nur nach 3× "open")

GEÄNDERTE DATEIEN:

  dino_wm/planning/cem.py (2 Änderungen):
    + _enforce_gripper_consistency()     — 57 Zeilen, neue Methode
    + Aufruf in plan() nach _quantize_gripper()  — 2 Zeilen

  isaacsim/.../planning_client.py (4 Änderungen):
    + Gripper State Machine + execute_gripper_action()  — 45 Zeilen
    + Episode-Reset (gripper_is_closed, open_command_count)  — 2 Zeilen
    + Offline-Loop: Gripper-Extraktion + Ausführung  — 8 Zeilen
    + Online-Loop: Gripper-Extraktion + Ausführung  — 8 Zeilen

ZUSAMMENHANG MIT VORHERIGEN PHASEN:

  Phase 19: CEM Action Bounds + Gripper-Quantisierung
            → Gripper wird binär quantisiert, aber temporal inkonsistent
  Phase 20a: EEF-Orientierung (preferred_joints)
  Phase 20b: EEF Hard-Lock Joint 6
            → EEF erreicht Würfel präzise, aber Gripper wird nie betätigt
  Phase 21: Gripper-Timing Fix (dieser Commit)
            → Gripper wird temporal konsistent UND tatsächlich ausgeführt

ERWARTETES VERHALTEN NACH FIX:

  Vorher:  EEF → Würfel ✓, Gripper NIE betätigt ✗
  Nachher: EEF → Würfel ✓, Gripper schließt bei Würfel ✓,
           Gripper bleibt geschlossen während Transport ✓
           (Hysterese verhindert versehentliches Öffnen)
```
