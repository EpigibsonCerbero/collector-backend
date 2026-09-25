#!/usr/bin/env bash
# Corre todos los casos de prueba del escáner contra el backend local.
# Uso: ./run_all.sh [caso]   (ej: ./run_all.sh 01_coche_suelto)
API="http://localhost:8000/api/v1/identify"
DIR="$(cd "$(dirname "$0")" && pwd)"

declare -A EXPECT
EXPECT[01_coche_suelto]="El embed debería matchear la 1era vista del catálogo (0.98+)."
EXPECT[02_angulo_diferente]="Multi-view debería matchear aunque el ángulo sea distinto (0.85+)."
EXPECT[03_poca_luz]="Con poca luz+blur: embed débil → needs_manual honesto."
EXPECT[04_imagen_ia]="Render IA: sin texto empaque → needs_manual o embed razonable."
EXPECT[05_multiples_piezas]="4 piezas en encuadre → debería identificar al menos 1 (la dominante)."
EXPECT[06_empaque_angulo]="Empaque: OCR del toy#/▶NNN debería ganar."

run_case() {
  local case_dir="$1"
  local name=$(basename "$case_dir")
  echo ""
  echo "═══════════════════════════════════════════════════════"
  echo "📁 Caso: $name"
  echo "   Esperado: ${EXPECT[$name]:-—}"
  echo "═══════════════════════════════════════════════════════"

  shopt -s nullglob
  for img in "$case_dir"/*.jpg "$case_dir"/*.jpeg "$case_dir"/*.png; do
    [ -f "$img" ] || continue
    echo ""
    echo "📷 $(basename "$img")"
    curl -s -m 30 -X POST "$API" -F "image=@$img" | python3 -c "
import json,sys
d=json.load(sys.stdin)
item=d.get('item') or {}
sug=d.get('suggestions') or []
top=sug[0] if sug else {}
print(f\"  method={d['method']}  conf={d['confidence']:.3f}  manual={d['needs_manual']}\")
print(f\"  item={item.get('name') or item.get('variant_name') or '—'}  toy={item.get('toy_number') or d.get('toy_number') or '—'}\")
print(f\"  top1={top.get('toy_number','?')} {top.get('name','?')[:45]} score={top.get('score',0):.3f}\")
print(f\"  timings={d.get('timings_ms')}\")
"
  done
}

if [ -n "$1" ]; then
  run_case "$DIR/$1"
else
  for d in "$DIR"/*/; do
    [ -d "$d" ] && [ "$(basename "$d")" != "__pycache__" ] && run_case "$d"
  done
fi

echo ""
echo "═══════════════════════════════════════════════════════"
echo "✅ Pruebas completadas."
echo "   Nota: caso 04 (imagen IA) necesita fotos de renders/gemini del usuario."
