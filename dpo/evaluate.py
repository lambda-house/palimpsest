#!/usr/bin/env python3
"""Generate completions from the model for human evaluation.

Outputs an HTML file with prompt + completion pairs for easy review.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


SYSTEM_PROMPT = (
    "Ты — писатель художественной прозы. Пиши в стиле, характерном для "
    "русской постмодернистской прозы: философские отступления, буддийские "
    "метафоры, сатира на современность, многослойная ирония, игра с "
    "поп-культурными отсылками."
)


def generate(model: str, prompt: str, max_tokens: int = 1024) -> str:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {
            "temperature": 0.8,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "num_predict": max_tokens,
        },
    })
    try:
        result = subprocess.run(
            ["curl", "-s", "http://localhost:11434/api/generate", "-d", payload],
            capture_output=True, text=True, timeout=600,
        )
        data = json.loads(result.stdout)
        return data.get("response", "").strip()
    except Exception as e:
        return f"[ERROR: {e}]"


def render_html(results: list, output_path: str):
    rows = []
    for i, r in enumerate(results, 1):
        prompt_html = r["prompt"].replace("&", "&amp;").replace("<", "&lt;")
        text_html = r["text"].replace("&", "&amp;").replace("<", "&lt;").replace("\n", "<br>")
        rows.append(f"""
        <div class="card" id="card-{i}">
            <div class="num">{i}/100</div>
            <div class="prompt">{prompt_html}</div>
            <div class="text">{text_html}</div>
            <div class="scores">
                <div class="score-row">
                    <label>Adherence (1-10): <input type="number" min="1" max="10" class="score" data-id="{i}" data-type="adherence"></label>
                    <label>Style (1-10): <input type="number" min="1" max="10" class="score" data-id="{i}" data-type="style"></label>
                </div>
                <label>Notes: <textarea class="note" data-id="{i}" style="width:100%;height:80px;resize:vertical;font-size:14px;padding:8px;box-sizing:border-box"></textarea></label>
            </div>
        </div>""")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Pelevin Nemo 12B - Evaluation</title>
<style>
body {{ font-family: Georgia, serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #f5f5f0; }}
h1 {{ text-align: center; }}
.card {{ background: white; border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin: 20px 0; }}
.num {{ color: #999; font-size: 12px; }}
.prompt {{ font-weight: bold; color: #333; margin: 10px 0; padding: 10px; background: #f0f0e8; border-radius: 4px; }}
.text {{ line-height: 1.6; color: #444; margin: 10px 0; white-space: pre-wrap; }}
.scores {{ margin-top: 15px; padding-top: 10px; border-top: 1px solid #eee; }}
.scores label {{ font-size: 13px; color: #666; display: block; margin: 5px 0; }}
.scores input[type=number] {{ width: 50px; }}
.score-row {{ display: flex; gap: 20px; align-items: center; }}
#export {{ position: fixed; bottom: 20px; right: 20px; padding: 12px 24px; background: #333; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 14px; }}
#export:hover {{ background: #555; }}
#stats {{ color: #666; margin: 10px 0; }}
.instructions {{ background: #fff8e1; border: 1px solid #ffe0b2; border-radius: 8px; padding: 20px; margin: 20px 0; line-height: 1.6; }}
.instructions h3 {{ margin-top: 0; }}
.instructions ul {{ margin: 8px 0; padding-left: 20px; }}
</style></head><body>
<h1>Pelevin Nemo 12B — Human Evaluation</h1>
<div class="instructions">
<h3>Как оценивать</h3>
<p>Каждая карточка содержит <b>задание</b> (что модель должна была написать) и <b>сгенерированный текст</b>. Оцените каждый по двум критериям:</p>
<ul>
<li><b>Adherence / Следование заданию (1-10)</b> — Соответствует ли текст заданию? Если просили про музыку — текст про музыку? Если просили диалог — есть диалог? Если указано конкретное место или ситуация — они присутствуют?
<br><i>1 = полностью игнорирует задание, 10 = точно следует заданию</i></li>
<li><b>Style / Стилистика (1-10)</b> — Похоже ли на Пелевина? Обращайте внимание на: ирония и многослойный подтекст, философские отступления, буддийские метафоры, сатира на современность, отсылки к поп-культуре, характерная лексика и ритм прозы.
<br><i>1 = безликая проза, 10 = неотличимо от оригинала</i></li>
</ul>
<p>В поле <b>Notes</b> отмечайте что угодно: удачные/неудачные места, фактические ошибки, повторы, путаницу персонажей, языковые проблемы.</p>
<p>Не думайте слишком долго — важно первое впечатление. Можно пропускать. Нажмите <b>Export Scores</b> когда закончите.</p>
</div>
{"".join(rows)}
<button id="export" onclick="exportScores()">Export Scores (JSON)</button>
<script>
function exportScores() {{
    const scores = [];
    for (let i = 1; i <= 100; i++) {{
        const adh = document.querySelector(`input[data-id="${{i}}"][data-type="adherence"]`);
        const sty = document.querySelector(`input[data-id="${{i}}"][data-type="style"]`);
        const note = document.querySelector(`textarea.note[data-id="${{i}}"]`);
        scores.push({{
            id: i,
            adherence: adh ? parseInt(adh.value) || null : null,
            style: sty ? parseInt(sty.value) || null : null,
            note: note ? note.value : ""
        }});
    }}
    const blob = new Blob([JSON.stringify(scores, null, 2)], {{type: "application/json"}});
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "evaluation_scores.json";
    a.click();
}}
</script></body></html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default="./dpo/prompts.jsonl")
    parser.add_argument("--model", default="pelevin-nemo")
    parser.add_argument("--output-json", default="./dpo/evaluation.jsonl")
    parser.add_argument("--output-html", default="./dpo/evaluation.html")
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    prompts = [json.loads(l) for l in Path(args.prompts).read_text().splitlines() if l.strip()]
    print(f"Generating {len(prompts)} completions from {args.model}...")

    results = []
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    # Write incrementally so results are available as they come
    with open(out_json, "w", encoding="utf-8") as f:
        for i, p in enumerate(prompts, 1):
            text = generate(args.model, p["prompt"], args.max_tokens)
            result = {"prompt": p["prompt"], "type": p.get("type", ""), "text": text}
            results.append(result)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            words = len(text.split())
            print(f"  [{i}/{len(prompts)}] {words} words | {p['prompt'][:60]}...", flush=True)

    # Render HTML
    render_html(results, args.output_html)
    print(f"\nResults: {args.output_json}")
    print(f"Review:  {args.output_html}")


if __name__ == "__main__":
    main()
