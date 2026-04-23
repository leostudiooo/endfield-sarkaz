#!/usr/bin/env python3
"""Clean arknights_sentences.txt for Sarkaz parallel corpus generation."""

import re
from pathlib import Path


def clean_corpus(input_path: str, output_path: str):
    with open(input_path, encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]

    dialogue_re = re.compile(r'^([^:]+?):\s+(.+)$')

    seen = set()
    cleaned = []
    stats = {
        'total': len(lines),
        'removed_weather': 0,
        'removed_location': 0,
        'removed_time_skip': 0,
        'removed_ellipsis': 0,
        'removed_dash': 0,
        'removed_short': 0,
        'removed_markup': 0,
        'removed_duplicate': 0,
        'kept_dialogue': 0,
        'kept_narration': 0,
    }

    for line in lines:
        # Subtitle / sticker markup
        if re.match(r'^【.+】$', line) or re.match(r'^\[\[.+\]\]$', line):
            stats['removed_markup'] += 1
            continue

        m = dialogue_re.match(line)
        if m:
            speaker, content = m.group(1), m.group(2).strip()

            # Pure ellipsis
            if re.match(r'^[.…]+$', content):
                stats['removed_ellipsis'] += 1
                continue
            # Pure dashes
            if re.match(r'^[—–\-]+$', content):
                stats['removed_dash'] += 1
                continue
            # Too short — ambiguous for cipher mapping
            if len(content) <= 2:
                stats['removed_short'] += 1
                continue

            text = content
            stats['kept_dialogue'] += 1
        else:
            # Non-dialogue line
            # Weather
            if '能见度' in line or re.match(r'^(晴|多云|阴|小雨|大雨|雪|雷)[\s\\]', line):
                stats['removed_weather'] += 1
                continue
            # Time skip
            if re.match(r'^.{0,4}(数日|次日|翌日|黄昏|深夜|清晨|黎明|傍晚|夜晚|正午|午后|数年|数月|同时|不久|此时|随后|稍后|之前|之后|片刻|不久后|数小时)', line):
                stats['removed_time_skip'] += 1
                continue
            # Location header — short, ends with location nouns
            if re.match(r'^[一-鿿A-Za-z0-9\s()（）]{1,12}(内|前|中|外|里|处|旁|边|下|上|间|入口|出口|大道|小路|走廊|大厅|房间|营地|要塞|都市|城市|城镇|村庄|据点|车站|港口|广场|顶端|底部|深处)$', line):
                stats['removed_location'] += 1
                continue
            # Time stamps like "2:19 P.M. 天气/晴"
            if re.match(r'^\d{1,2}:\d{2}\s', line):
                stats['removed_weather'] += 1
                continue

            # Narrative text — keep as-is
            text = line
            stats['kept_narration'] += 1

        # Global minimum length — too short for meaningful cipher mapping
        if len(text) <= 2:
            stats['removed_short'] += 1
            continue
        # Pure dashes (may come from narration lines)
        if re.match(r'^[—–\-]+$', text):
            stats['removed_dash'] += 1
            continue

        # Dedup
        if text in seen:
            stats['removed_duplicate'] += 1
            continue
        seen.add(text)
        cleaned.append(text)

    # Write output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in cleaned:
            f.write(line + '\n')

    # Report
    total_kept = len(cleaned)
    total_removed = stats['total'] - total_kept
    print(f"=== Cleaning Report ===")
    print(f"Input:  {stats['total']:>8,} lines")
    print(f"Output: {total_kept:>8,} lines  ({total_removed:,} removed, {total_removed/stats['total']*100:.1f}%)")
    print()
    print("Removed by category:")
    for k, v in stats.items():
        if k.startswith('removed') and v > 0:
            print(f"  {k.replace('removed_', ''):>12s}: {v:>7,}")
    print()
    print("Kept:")
    print(f"  {'dialogue':>12s}: {stats['kept_dialogue']:>7,}")
    print(f"  {'narration':>12s}: {stats['kept_narration']:>7,}")
    print(f"  {'deduped':>12s}: {stats['removed_duplicate']:>7,}")


if __name__ == '__main__':
    clean_corpus(
        'corpus/raw/ak/arknights_sentences.txt',
        'corpus/raw/ak/arknights_cleaned.txt',
    )
