#!/usr/bin/env bash
# Prueba el endpoint /identify con una o varias imágenes.
#   ./test_identify.sh /tmp/mgt00726.jpg ~/Fotos/gemini.jpg
API="http://192.168.100.24:8000/api/v1/identify"
[ $# -eq 0 ] && { echo "Uso: $0 <imagen> [imagen...]"; exit 1; }
for img in "$@"; do
  [ -f "$img" ] || { echo "— no existe: $img"; continue; }
  echo "════════════════════════════════════════"
  echo "📷 $img"
  curl -s -m 60 -X POST "$API" -F "image=@$img" | python3 -c "
import json,sys
d=json.load(sys.stdin)
print('  method  :', d['method'], '| needs_manual:', d['needs_manual'], '| conf:', round(d['confidence'],3))
item=d.get('item') or {}
print('  item    :', item.get('name') or item.get('variant_name') or '—')
print('  toy     :', item.get('toy_number') or d.get('toy_number') or '—')
print('  ocr     :', repr((d.get('debug') or {}).get('ocr_preview',''))[:90])
print('  top1    :', d['suggestions'][0]['toy_number'], d['suggestions'][0]['name'][:50], d['suggestions'][0]['score'] if d['suggestions'] else '—')
print('  timings :', d.get('timings_ms'))
"
done
