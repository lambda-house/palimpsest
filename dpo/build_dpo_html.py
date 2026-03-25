#!/usr/bin/env python3
"""Build HTML review page for DPO preference scoring."""

import json
import re
from pathlib import Path

def clean_text(text):
    text = re.sub(r'<SPECIAL_\d+>', '', text)
    prefixes = ['Передайте','Добавьте','Подчеркните','Подумайте','Опишите',
                'Напишите','Представьте','Создайте','Включите','Используйте',
                'Покажите','Расскажите','Сделайте','Обратите','Уделите',
                'Пирожок','Пишите']
    for prefix in prefixes:
        if text.strip().startswith(prefix):
            m = re.match(r'^[^.!?\n]+[.!?]\s*', text.strip())
            if m:
                text = text.strip()[m.end():]
            break
    return text.strip()

def build_html(input_path, output_path, title="DPO Preference Scoring"):
    data = [json.loads(l) for l in Path(input_path).read_text().splitlines()]

    rows = []
    for i, item in enumerate(data, 1):
        prompt = item['prompt'].replace('&', '&amp;').replace('<', '&lt;')
        comps = item.get('completions', [])
        n = len(comps)

        comps_html = ''
        for j, c in enumerate(comps, 1):
            text = clean_text(c['text'])
            text_html = text[:600].replace('&', '&amp;').replace('<', '&lt;').replace('\n', '<br>')
            temp = c.get('temperature', '?')
            comps_html += (
                f'<div class="completion">'
                f'<div class="comp-header">Variant {j} (temp={temp})</div>'
                f'<div class="comp-text">{text_html}...</div>'
                f'</div>'
            )

        rows.append(f"""
        <div class="card" id="card-{i}">
            <div class="num">{i}/{len(data)}</div>
            <div class="prompt">{prompt}</div>
            {comps_html}
            <div class="scores">
                <div class="score-row">
                    <label>Best # (1-{n}): <input type="number" min="1" max="{n}" class="score" data-id="{i}" data-type="best"></label>
                    <label>Worst # (1-{n}): <input type="number" min="1" max="{n}" class="score" data-id="{i}" data-type="worst"></label>
                </div>
                <label>Notes: <textarea class="note" data-id="{i}" style="width:100%;height:60px;resize:vertical;font-size:14px;padding:8px;box-sizing:border-box"></textarea></label>
            </div>
        </div>""")

    total = len(data)
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
body {{ font-family: Georgia, serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #f5f5f0; }}
h1 {{ text-align: center; }}
.card {{ background: white; border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin: 20px 0; }}
.num {{ color: #999; font-size: 12px; }}
.prompt {{ font-weight: bold; color: #333; margin: 10px 0; padding: 10px; background: #f0f0e8; border-radius: 4px; }}
.completion {{ margin: 10px 0; padding: 10px; border-left: 3px solid #ddd; }}
.comp-header {{ font-size: 12px; color: #888; margin-bottom: 5px; }}
.comp-text {{ line-height: 1.5; color: #444; font-size: 14px; }}
.scores {{ margin-top: 15px; padding-top: 10px; border-top: 1px solid #eee; }}
.score-row {{ display: flex; gap: 20px; align-items: center; margin-bottom: 8px; }}
.scores label {{ font-size: 13px; color: #666; }}
.scores input[type=number] {{ width: 50px; }}
.instructions {{ background: #fff8e1; border: 1px solid #ffe0b2; border-radius: 8px; padding: 20px; margin: 20px 0; line-height: 1.6; }}
#export {{ position: fixed; bottom: 20px; right: 20px; padding: 12px 24px; background: #333; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 14px; }}
#export:hover {{ background: #555; }}
</style></head><body>
<h1>{title}</h1>
<div class="instructions">
<h3>Как оценивать</h3>
<p>Для каждого промпта показаны варианты текста при разных температурах. Выберите:</p>
<ul>
<li><b>Best #</b> -- номер лучшего варианта (следует заданию + хороший стиль)</li>
<li><b>Worst #</b> -- номер худшего варианта (не по теме, плоский стиль, повторы)</li>
</ul>
<p>Пропускайте если все варианты одинаковые. Нажмите <b>Export</b> когда закончите.</p>
</div>
{"".join(rows)}
<button id="export" onclick="exportScores()">Export Preferences (JSON)</button>
<script>
function exportScores() {{
    const scores = [];
    for (let i = 1; i <= {total}; i++) {{
        const best = document.querySelector(`input[data-id="${{i}}"][data-type="best"]`);
        const worst = document.querySelector(`input[data-id="${{i}}"][data-type="worst"]`);
        const note = document.querySelector(`textarea[data-id="${{i}}"]`);
        scores.push({{
            id: i,
            best: best ? parseInt(best.value) || null : null,
            worst: worst ? parseInt(worst.value) || null : null,
            note: note ? note.value : ""
        }});
    }}
    const blob = new Blob([JSON.stringify(scores, null, 2)], {{type: "application/json"}});
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "preferences.json";
    a.click();
}}
</script></body></html>"""

    Path(output_path).write_text(html, encoding='utf-8')
    print(f"Built {output_path} with {total} prompts")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--title', default='DPO Preference Scoring')
    args = parser.parse_args()
    build_html(args.input, args.output, args.title)
