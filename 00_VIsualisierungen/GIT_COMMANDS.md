# 📚 Git Commands Referenz

> Dokumentation der Git-Befehle für das DINO World Model Projekt

---

## 📑 Inhaltsverzeichnis

1. [Grundlegende Konzepte](#1-grundlegende-konzepte)
2. [Verwendete Befehle beim Merge](#2-verwendete-befehle-beim-merge)
3. [Häufige Git-Workflows](#3-häufige-git-workflows)
4. [Konfliktlösung](#4-konfliktlösung)
5. [Nützliche Befehle](#5-nützliche-befehle)

---

## 1. Grundlegende Konzepte

### 1.1 Git Bereiche

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          GIT BEREICHE                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐  │
│  │  Working     │    │   Staging    │    │    Local     │    │  Remote  │  │
│  │  Directory   │───►│    Area      │───►│    Repo      │───►│   Repo   │  │
│  │              │    │   (Index)    │    │              │    │ (origin) │  │
│  └──────────────┘    └──────────────┘    └──────────────┘    └──────────┘  │
│         │                  │                   │                   │        │
│         │    git add       │    git commit     │     git push      │        │
│         ├─────────────────►├──────────────────►├──────────────────►│        │
│         │                  │                   │                   │        │
│         │                  │                   │     git fetch     │        │
│         │                  │                   │◄──────────────────┤        │
│         │                  │                   │                   │        │
│         │                  │    git checkout   │                   │        │
│         │◄─────────────────┴───────────────────┤                   │        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Branch-Divergenz

```
Was passiert wenn Branches divergieren:

          A---B---C  (origin/main - Remote)
         /
    D---E---F---G    (main - Lokal)

"Your branch and 'origin/main' have diverged,
 and have 1 and 1 different commits each"
 
 → 1 Commit auf Remote (C), 1 Commit lokal (G)
```

---

## 2. Verwendete Befehle beim Merge

### 2.1 `git status`

```bash
git status
```

**Was es macht:**
- Zeigt den aktuellen Zustand des Repositories
- Listet modifizierte, staged und untracked Dateien
- Zeigt Branch-Information (ahead/behind von Remote)

**Output-Erklärung:**
```
On branch main
Your branch and 'origin/main' have diverged,    ← Branches sind unterschiedlich
and have 1 and 1 different commits each.        ← Jeweils 1 Commit verschieden

Changes not staged for commit:                   ← Geändert, aber nicht staged
        modified:   conf/train.yaml

Untracked files:                                 ← Neue Dateien, nicht in Git
        DINO_WM_PLANNING_DOCUMENTATION.md
        env/franka_cube_stack/
```

---

### 2.2 `git stash`

```bash
git stash -u -m "WIP: Planning Wrapper und Dokumentation"
```

**Was es macht:**
- Speichert alle lokalen Änderungen temporär auf einem "Stapel"
- Macht das Working Directory sauber (wie nach frischem Clone)
- `-u` = auch **u**ntracked files (neue Dateien) mit stashen
- `-m "..."` = Beschreibung für den Stash

**Warum benötigt:**
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  PROBLEM: Du hast lokale Änderungen UND willst Remote-Änderungen holen      │
│                                                                             │
│  Lokale Änderungen          Remote-Änderungen                               │
│  ┌──────────────┐           ┌──────────────┐                                │
│  │ train.yaml   │           │ train.yaml   │  ← Gleiche Datei!              │
│  │ (geändert)   │           │ (geändert)   │                                │
│  └──────────────┘           └──────────────┘                                │
│                                                                             │
│  git pull würde fehlschlagen: "Please commit or stash your changes"         │
│                                                                             │
│  LÖSUNG: Stash = Temporär weglegen → Pull → Stash zurückholen              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Stash-Befehle:**
```bash
git stash list              # Alle Stashes anzeigen
git stash show              # Inhalt des letzten Stash zeigen
git stash pop               # Letzten Stash anwenden UND löschen
git stash apply             # Letzten Stash anwenden, ABER behalten
git stash drop              # Letzten Stash löschen
git stash clear             # ALLE Stashes löschen
```

---

### 2.3 `git pull --rebase`

```bash
git pull --rebase origin main
```

**Was es macht:**
1. `git fetch origin main` - Holt Remote-Änderungen
2. `git rebase origin/main` - Setzt lokale Commits auf Remote-Stand

**Unterschied: Merge vs. Rebase**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  VORHER (Branches divergiert):                                              │
│                                                                             │
│          A---B---C  (origin/main)                                           │
│         /                                                                   │
│    D---E---F---G    (main lokal)                                           │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  NACH `git pull` (Standard = Merge):                                        │
│                                                                             │
│          A---B---C                                                          │
│         /         \                                                         │
│    D---E---F---G---M    (M = Merge-Commit)                                 │
│                                                                             │
│  → Erstellt extra Merge-Commit                                              │
│  → Historie wird "verzweigt"                                                │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  NACH `git pull --rebase`:                                                  │
│                                                                             │
│    D---E---A---B---C---G'   (G' = G neu angewendet)                        │
│                                                                             │
│  → Kein Merge-Commit                                                        │
│  → Lineare, saubere Historie                                                │
│  → Lokaler Commit G wird "neu geschrieben" als G'                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Wann was verwenden:**
| Situation | Empfehlung |
|-----------|------------|
| Eigene lokale Änderungen | `--rebase` (saubere Historie) |
| Shared Branch mit Team | `merge` (Historie erhalten) |
| Komplexe Merge-Konflikte | `merge` (einfacher abzubrechen) |

---

### 2.4 `git stash pop`

```bash
git stash pop
```

**Was es macht:**
1. Wendet den letzten Stash auf das Working Directory an
2. Löscht den Stash (wenn erfolgreich)
3. Bei Konflikten: Stash bleibt erhalten

**Mögliche Outputs:**

```bash
# Erfolg (kein Konflikt):
Dropped refs/stash@{0} (abc123...)

# Mit Konflikt:
Auto-merging conf/train.yaml
CONFLICT (content): Merge conflict in conf/train.yaml
The stash entry is kept in case you need it again.
```

---

### 2.5 Konfliktlösung

```bash
# 1. Konflikt-Datei bearbeiten (<<<<<<< und >>>>>>> entfernen)
# 2. Als gelöst markieren:
git add conf/train.yaml

# 3. Stash manuell löschen (da pop bei Konflikt nicht löscht):
git stash drop
```

**Konflikt-Marker in Datei:**
```yaml
encoder_lr: 1e-6
<<<<<<< Updated upstream
  decoder_lr: 1e-4   # von 3e-4      ← Remote-Version
  predictor_lr: 2e-4 # von 5e-4
=======
  decoder_lr: 1e-4                    ← Lokale Version
  predictor_lr: 2e-4
>>>>>>> Stashed changes
```

**Bedeutung:**
- `<<<<<<< Updated upstream` = Beginn Remote-Version
- `=======` = Trenner
- `>>>>>>> Stashed changes` = Ende lokale Version

**Lösung:** Entscheide welche Version (oder Kombination) du willst, lösche die Marker.

---

### 2.6 `git add`

```bash
git add DINO_WM_PLANNING_DOCUMENTATION.md env/franka_cube_stack/
```

**Was es macht:**
- Fügt Dateien zur **Staging Area** hinzu
- Staging Area = "Vorbereitungsbereich" für nächsten Commit
- Kann einzelne Dateien, Ordner oder Patterns sein

**Varianten:**
```bash
git add .                    # Alle Änderungen im aktuellen Ordner
git add -A                   # Alle Änderungen im ganzen Repo
git add *.py                 # Alle Python-Dateien
git add -p                   # Interaktiv einzelne Änderungen auswählen
```

---

### 2.7 `git commit`

```bash
git commit -m "feat(planning): Add FrankaCubeStackWrapper..."
```

**Was es macht:**
- Erstellt einen neuen Commit mit allen staged Änderungen
- `-m "..."` = Commit-Message direkt angeben
- Ohne `-m`: Öffnet Editor für längere Message

**Commit-Message Konventionen:**
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
| Type | Beschreibung |
|------|--------------|
| `feat` | Neues Feature |
| `fix` | Bugfix |
| `docs` | Dokumentation |
| `refactor` | Code-Umbau ohne Funktionsänderung |
| `test` | Tests hinzufügen/ändern |
| `chore` | Maintenance (Dependencies, Config) |

---

## 3. Häufige Git-Workflows

### 3.1 Feature entwickeln und pushen

```bash
# 1. Neuesten Stand holen
git pull --rebase origin main

# 2. Änderungen machen
# ... edit files ...

# 3. Status prüfen
git status

# 4. Änderungen stagen
git add .

# 5. Committen
git commit -m "feat: Add new feature"

# 6. Pushen
git push origin main
```

### 3.2 Änderungen verwerfen

```bash
# Einzelne Datei zurücksetzen (unstaged):
git checkout -- conf/train.yaml

# Alle unstaged Änderungen verwerfen:
git checkout -- .

# Staged Änderungen unstagen:
git restore --staged conf/train.yaml

# Letzten Commit rückgängig (Änderungen behalten):
git reset --soft HEAD~1

# Letzten Commit komplett verwerfen:
git reset --hard HEAD~1
```

### 3.3 Branches

```bash
# Neuen Branch erstellen und wechseln:
git checkout -b feature/new-wrapper

# Branch wechseln:
git checkout main

# Branch löschen:
git branch -d feature/new-wrapper

# Alle Branches anzeigen:
git branch -a
```

---

## 4. Konfliktlösung

### 4.1 Workflow bei Konflikten

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     KONFLIKT-LÖSUNGS-WORKFLOW                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. Konflikt tritt auf (nach pull/merge/stash pop)                         │
│     └── Git zeigt: "CONFLICT (content): Merge conflict in file.txt"        │
│                                                                             │
│  2. Konflikt-Dateien identifizieren                                         │
│     └── git status                                                          │
│     └── "Unmerged paths:" zeigt Konflikt-Dateien                           │
│                                                                             │
│  3. Datei öffnen und Konflikt lösen                                         │
│     └── Suche nach <<<<<<< und >>>>>>>                                      │
│     └── Entscheide welche Version                                          │
│     └── Lösche die Marker                                                  │
│                                                                             │
│  4. Als gelöst markieren                                                    │
│     └── git add <konflikt-datei>                                           │
│                                                                             │
│  5. Weiter mit ursprünglicher Operation                                     │
│     └── Bei merge: git commit                                              │
│     └── Bei rebase: git rebase --continue                                  │
│     └── Bei stash: git stash drop                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Konflikt abbrechen

```bash
# Merge abbrechen:
git merge --abort

# Rebase abbrechen:
git rebase --abort

# Bei Stash-Konflikt: Änderungen verwerfen, Stash behalten
git checkout -- .
```

---

## 5. Nützliche Befehle

### 5.1 Informationen

```bash
# Commit-Historie anzeigen:
git log --oneline -10

# Änderungen einer Datei anzeigen:
git diff conf/train.yaml

# Staged Änderungen anzeigen:
git diff --staged

# Wer hat welche Zeile geändert:
git blame conf/train.yaml

# Remote-URLs anzeigen:
git remote -v
```

### 5.2 Rückgängig machen

```bash
# Letzte Commit-Message ändern:
git commit --amend -m "Neue Message"

# Datei aus letztem Commit entfernen (behalten im Working Dir):
git reset HEAD~1 -- datei.txt

# Zu bestimmtem Commit zurück (GEFÄHRLICH - löscht Historie):
git reset --hard <commit-hash>
```

### 5.3 Aufräumen

```bash
# Untracked files anzeigen die gelöscht würden:
git clean -n

# Untracked files löschen:
git clean -f

# Auch Ordner löschen:
git clean -fd

# Lokale Branches die nicht mehr auf Remote existieren löschen:
git fetch --prune
git branch -vv | grep 'gone]' | awk '{print $1}' | xargs git branch -d
```

### 5.4 Aliase (Abkürzungen)

```bash
# In ~/.gitconfig oder git config --global:
git config --global alias.st status
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.lg "log --oneline --graph --all"
```

---

## Zusammenfassung: Der Merge-Vorgang

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  WAS WIR GEMACHT HABEN:                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. git status                                                              │
│     → Gesehen: Branches divergiert, lokale Änderungen vorhanden            │
│                                                                             │
│  2. git stash -u -m "WIP: Planning Wrapper"                                │
│     → Lokale Änderungen temporär gesichert                                 │
│     → Working Directory ist jetzt sauber                                   │
│                                                                             │
│  3. git pull --rebase origin main                                          │
│     → Remote-Änderungen geholt                                             │
│     → Lokale Commits auf neuen Stand "rebased"                             │
│     → Lineare Historie erstellt                                            │
│                                                                             │
│  4. git stash pop                                                          │
│     → Gesicherte Änderungen zurückgeholt                                   │
│     → KONFLIKT in train.yaml aufgetreten                                   │
│                                                                             │
│  5. Konflikt manuell gelöst                                                │
│     → <<<<<<< und >>>>>>> Marker entfernt                                  │
│     → Gewünschte Version behalten                                          │
│                                                                             │
│  6. git add conf/train.yaml                                                │
│     → Konflikt als gelöst markiert                                         │
│                                                                             │
│  7. git stash drop                                                         │
│     → Stash gelöscht (war bei Konflikt noch vorhanden)                     │
│                                                                             │
│  8. git add ... && git commit                                              │
│     → Alle Änderungen committet                                            │
│                                                                             │
│  9. git push (ausstehend)                                                  │
│     → Änderungen auf Remote pushen                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

*Dokumentation erstellt am 01.02.2026*
