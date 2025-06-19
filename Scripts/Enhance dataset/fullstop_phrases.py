#!/usr/bin/env python3
"""
build_music_sentences.py  –  keep your parallel-array LANG_DATA

• Reads two JSONL files:
    1) dataset.jsonl  (your rows with wikidata_id & entity_types)
    2) labels.jsonl   (one row per Q-ID: {"id": "...", "source": "...", "target": "..."} )
• Emits output.jsonl with only the Musical-work rows transformed into your
  randomized “I like listening to…” sentence pairs.
"""

import argparse
import json
import random
import re
import uuid
from pathlib import Path
from typing import Dict, List

# ──────────────────────────────────────────────────────────────────────────────
# 1.  ENGLISH WORD VECTORS  (indices drive the choice)
# ──────────────────────────────────────────────────────────────────────────────
CONNECTORS_EN: List[str] = ["while", "when"]

ACTIVITIES_EN: List[str] = [
    "cleaning", "working", "working out", "relaxing", "reading", "focusing",
    "at a party", "eating", "driving", "showering", "sleeping", "drawing",
    "walking", "napping", "gaming", "cleaning", "crying",
]

# ──────────────────────────────────────────────────────────────────────────────
# 2.  TARGET-LANGUAGE WORD VECTORS
# ──────────────────────────────────────────────────────────────────────────────
LANG_DATA: Dict[str, Dict[str, List[str]]] = {
    "de": {
        "openers":   ["Ich höre gerne"],
        "connectors": ["während", "wenn"],
        "activities": [
                "putze", "arbeite", "trainiere", "mich entspanne", "lese",
                "mich konzentriere", "auf einer Party bin", "esse",
                "Auto fahre", "dusche", "schlafe", "zeichne", "spazieren gehe",
                "ein Nickerchen mache", "spiele", "putze", "weine"
            ],
    },
    "it": {
        "openers":   ["Mi piace ascoltare", "Ascolto"],
        "connectors": ["mentre", "quando"],
        "activities": [
            "pulendo", "lavorando", "allenandomi", "rilassandomi", "leggendo",
            "concentrandomi", "a una festa", "mangiando", "guidando",
            "facendo la doccia", "dormendo", "disegnando", "camminando",
            "facendo un pisolino", "giocando", "pulendo", "piangendo",
        ],
    },
    "es": {
        "openers":   ["Me gusta escuchar", "Escucho"],
        "connectors": ["mientras", "cuando"],
        "activities": [
            "limpiando", "trabajando", "poniéndome en forma", "relajándome",
            "leyendo", "concentrándome", "en una fiesta", "comiendo",
            "conduciendo", "duchándome", "durmiendo", "dibujando",
            "caminando", "echando una siesta", "jugando", "limpiando",
            "llorando",
        ],
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# 3.  HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def load_label_map(path: Path) -> Dict[str, Dict[str, str]]:
    """
    Load Q-ID → { "en": source_label, "target": target_label }.
    """
    m: Dict[str, Dict[str, str]] = {}
    with path.open(encoding="utf8") as f:
        for line in f:
            row = json.loads(line)
            m[row["id"]] = {
                "en": row.get("source", "") or "",
                "target": row.get("target", "") or ""
            }
    return m


def find_qid(row: Dict) -> str:
    """
    Get the Q-ID: prefer wikidata_id, then entities[], then leading Q… in id.
    """
    if "wikidata_id" in row:
        return row["wikidata_id"]
    if "entities" in row and row["entities"]:
        return row["entities"][0]
    m = re.match(r"(Q\d+)", row.get("id", ""))
    if m:
        return m.group(1)
    raise ValueError(f"No Q-ID in row {row!r}")


def build_pair(
    row: Dict,
    label_map: Dict[str, Dict[str, str]]
) -> Dict:
    """
    Build your bilingual sentence pair for one Musical work row.
    """
    tgt_locale = row["target_locale"]
    if tgt_locale not in LANG_DATA:
        raise ValueError(f"Unsupported language: {tgt_locale}")

    vectors = LANG_DATA[tgt_locale]

    # 1) pick random connector & activity indices
    ci = random.randrange(len(CONNECTORS_EN))
    ai = random.randrange(len(ACTIVITIES_EN))

    conn_en = CONNECTORS_EN[ci]
    conn_tgt = vectors["connectors"][ci]

    act_en  = ACTIVITIES_EN[ai]
    act_tgt = vectors["activities"][ai]

    opener = random.choice(vectors["openers"])

    # 2) lookup labels from your mapping file
    qid    = find_qid(row)
    labs   = label_map.get(qid, {"en": "", "target": ""})
    work_en = labs["en"] or qid
    work_tgt = labs["target"] or qid

    # 3) assemble sentences
    source = f"I like listening to {work_en} {conn_en} I'm {act_en}."
    #target = f"{opener} {work_tgt} {conn_tgt} sto {act_tgt}."
    target = f"{opener} {work_tgt} {conn_tgt} estoy {act_tgt}."

    return {
        "id":            f"{qid}_{uuid.uuid4().hex[:8]}",
        "source_locale": "en",
        "target_locale": tgt_locale,
        "source":        source,
        "target":        target,
        "entities":      [qid],
        "from":          "generated",
    }


def process(dataset_path: Path, label_map_path: Path, out_path: Path) -> None:
    label_map = load_label_map(label_map_path)
    used_ids = set()

    with dataset_path.open(encoding="utf8") as fin, out_path.open("w", encoding="utf8") as fout:

        for ln, line in enumerate(fin, 1):
            row = json.loads(line)

            # only handle musical works
            if "entity_types" not in row or "Musical work" not in row["entity_types"]:
                continue
            if "wikidata_id" in row and row["wikidata_id"] in used_ids:
                continue
            try:
                pair = build_pair(row, label_map)
                fout.write(json.dumps(pair, ensure_ascii=False) + "\n")
                used_ids.add(row.get("wikidata_id"))
            except Exception as e:
                print(f"[line {ln}] skipped: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build bilingual music-listening sentences from Musical work rows"
    )
    parser.add_argument("dataset", type=Path,
                        help="Your input JSONL with wikidata_id & entity_types")
    parser.add_argument("labels", type=Path,
                        help="Q-ID → {source, target} JSONL mapping file")
    parser.add_argument("-o", "--output", type=Path,
                        default=Path("output.jsonl"),
                        help="Where to write the new sentence pairs")
    args = parser.parse_args()

    process(args.dataset, args.labels, args.output)
    print(f"Done → {args.output}")
