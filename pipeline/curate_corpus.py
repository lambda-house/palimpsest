#!/usr/bin/env python3
"""Curate author corpora to essential works only, removing duplicates and non-fiction."""

from pathlib import Path
import shutil

def curate(corpus_dir, keep_patterns, output_dir=None):
    """Keep only files matching patterns, skip duplicates (_2, _3 etc)."""
    src = Path(corpus_dir)
    dst = Path(output_dir) if output_dir else src

    all_files = sorted(src.glob('*.txt'))
    kept = []

    for f in all_files:
        name = f.stem
        # Skip obvious duplicates
        if any(name.endswith(s) for s in ['_2','_3','_4','_5','_6','_7','_8']):
            continue
        # Skip collections/compilations
        skip_words = ['собрание', 'том ', 'сборник', 'избранн', 'полное',
                      'весь ', 'в одном томе', 'в 8 томах', 'в 10 томах',
                      'в десяти томах', 'в одиннадцати', 'в 14 томах',
                      'OFF-LINE', 'интервью', 'письма', 'дневник',
                      'неизвестные', 'черновики', 'рукописи', 'переводы',
                      'публицистика', 'комментарии', 'хронология',
                      'неопубликованное', 'энциклопедия', 'стенограмма',
                      'рабочие дневники', 'что такое фантастика',
                      'мой бедный', 'чаша жизни', 'статьи']
        if any(w in name.lower() for w in skip_words):
            continue
        # Check against keep patterns if provided
        if keep_patterns:
            matched = any(p.lower() in name.lower() for p in keep_patterns)
            if not matched:
                continue

        wc = len(f.read_text().split())
        if wc < 3000:  # Skip tiny fragments
            continue
        kept.append((f, wc))

    return kept


# === BULGAKOV ===
print("=== BULGAKOV ===")
bulgakov_keep = [
    'Мастер и Маргарита', 'Белая гвардия', 'Собачье сердце',
    'Роковые яйца', 'Дьяволиада', 'Театральный роман',
    'Записки покойника', 'Жизнь господина де Мольера',
    'Бег', 'Дни Турбиных', 'Записки юного врача', 'Морфий',
    'Иван Васильевич', 'Зойкина квартира', 'Кабала святош',
    'Багровый остров', 'Адам и Ева', 'Записки на манжетах',
    'Блаженство', 'Александр Пушкин', 'Последние дни',
]
kept = curate('/Users/timur/work/pelevin/corpus/clean/bulgakov', bulgakov_keep)
total = sum(wc for _, wc in kept)
print(f"Kept {len(kept)} files, {total:,} words")
for f, wc in sorted(kept, key=lambda x: -x[1]):
    print(f"  {wc:>8,}  {f.name}")

# === STRUGATSKY ===
print("\n=== STRUGATSKY ===")
strugatsky_keep = [
    'Пикник на обочине', 'Трудно быть богом', 'Понедельник начинается',
    'Обитаемый остров', 'Жук в муравейнике', 'Волны гасят ветер',
    'За миллиард лет', 'Улитка на склоне', 'Град обреченный',
    'Хищные вещи века', 'Далекая радуга', 'Полдень',
    'Стажеры', 'Страна багровых туч', 'Попытка к бегству',
    'Второе нашествие марсиан', 'Сказка о тройке', 'Сказка о Тройке',
    'Малыш', 'Парень из преисподней', 'Гадкие лебеди',
    'Хромая судьба', 'Отягощенные злом', 'Бессильные мира сего',
    'Отель', 'Повесть о дружбе',
]
kept2 = curate('/Users/timur/work/pelevin/corpus/clean/strugatsky', strugatsky_keep)
total2 = sum(wc for _, wc in kept2)
print(f"Kept {len(kept2)} files, {total2:,} words")
for f, wc in sorted(kept2, key=lambda x: -x[1]):
    print(f"  {wc:>8,}  {f.name}")
