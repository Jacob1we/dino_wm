#!/usr/bin/env bash
# Sanity-Check für ALLE trainierten Modelle unter outputs/
# Output: .../plan_outputs/Sanity_Check/model_yymmdd_hh-mm/
# Überspringt Modelle deren Dataset nicht mehr existiert.

OUTPUTS_DIR="/home/tsp_jw/Desktop/dino_wm/outputs"
SANITY_DIR="/home/tsp_jw/Desktop/isaacsim/00_Franka_Cube_Stack/Franka_Cube_Stacking/plan_outputs/Sanity_Check"
SCRIPT="/home/tsp_jw/Desktop/dino_wm/wm_sanity_check.py"

mkdir -p "$SANITY_DIR"

total=0
success=0
fail=0

find "$OUTPUTS_DIR" -name "model_latest.pth" -type f | sed "s|${OUTPUTS_DIR}/||;s|/checkpoints/model_latest.pth||" | sort | while read -r dir; do
  # Convert to model_yymmdd_hh-mm
  if echo "$dir" | grep -qE '^[0-9]{4}-'; then
    yy=$(echo "$dir" | cut -c3-4)
    mm=$(echo "$dir" | cut -c6-7)
    dd=$(echo "$dir" | cut -c9-10)
    hh=$(echo "$dir" | cut -d'/' -f2 | cut -c1-2)
    mi=$(echo "$dir" | cut -d'/' -f2 | cut -c4-5)
    out_name="model_${yy}${mm}${dd}_${hh}-${mi}"
  else
    part1=$(echo "$dir" | cut -d'/' -f1)
    part2=$(echo "$dir" | cut -d'/' -f2)
    out_name="model_${part1}_${part2}"
  fi

  out_path="${SANITY_DIR}/${out_name}"

  # Skip wenn bereits vorhanden
  if [[ -f "${out_path}/metrics.json" ]]; then
    echo "SKIP (existiert): ${out_name}  ← ${dir}"
    continue
  fi

  # Skip wenn Dataset nicht mehr existiert
  cfg_file="${OUTPUTS_DIR}/${dir}/hydra.yaml"
  dpath=$(grep "data_path:" "$cfg_file" 2>/dev/null | head -1 | awk '{print $2}')
  if [[ -n "$dpath" ]] && [[ ! -d "$dpath" ]]; then
    echo "SKIP (Dataset fehlt): ${out_name}  ← ${dir}  (${dpath})"
    continue
  fi

  echo ""
  echo "================================================================"
  echo "  ${out_name}  ← ${dir}"
  echo "================================================================"

  if conda run -n dino_wm python "$SCRIPT" \
      --model_name "$dir" \
      --output_dir "$out_path" \
      --n_episodes 5 \
      --rollout_len 5; then
    echo "✅ OK: ${out_name}"
  else
    echo "❌ FEHLER: ${out_name} (${dir})"
  fi
done

echo ""
echo "========================================"
echo "Alle Sanity-Checks abgeschlossen."
echo "Output: ${SANITY_DIR}"
echo "========================================"
