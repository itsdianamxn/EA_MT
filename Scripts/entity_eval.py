import json
import os
import re
from typing import Dict, List, Set

ENTITY_TYPES: List[str] = [
    "Musical work", "Artwork", "Food", "Animal", "Plant", "Book",
    "Book series", "Fictional entity", "Landmark", "Movie",
    "Place of worship", "Natural place", "TV series", "Person",
]

TARGET_LANGUAGE   = "Fr" #for print
#PATH_TO_REFERENCES = r"C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/data/predictions_for_comparison/fr_FR.jsonl"  # gold file
PATH_TO_REFERENCES = r"C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/data/predictions_for_comparison/it_IT.jsonl"  # gold file
#PATH_TO_PREDICTIONS = r"C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/data/predictions/trained_t5base_model/results_base.jsonl"
#PATH_TO_PREDICTIONS = "C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/data/predictions/curriculum_training/t5_extended_dataset/rbe_it_new.jsonl"
#PATH_TO_PREDICTIONS = "C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/data/predictions/trained_mt5_model/double_extended_base/results_it.jsonl"
PATH_TO_PREDICTIONS = "C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/Scripts/Results/results_it_t5_t5_enti_out.jsonl"
#PATH_TO_PREDICTIONS = "C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/data/predictions_for_comparison/de_DE.jsonl"
#PATH_TO_PREDICTIONS = "C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/data/predictions/trained_mt5_model/extended/results_ar.jsonl"

#PATH_TO_PREDICTIONS = "C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/data/predictions/curriculum_training/t5_extended_dataset/results_es.jsonl"
#PATH_TO_PREDICTIONS = "C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/data/predictions/trained_mt5_model/extended/results_ar.jsonl"
#PATH_TO_PREDICTIONS = "C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/data/predictions/trained_mt5_model/extended/results_de.jsonl"

VERBOSE =True
_QID_RE   = re.compile(r"Q\d+")
_IDX_RE   = re.compile(r"Q\d+_\d+")

def _instance_id(rec: dict) -> str:

    if rec.get("wikidata_id"):
        return _QID_RE.search(rec["wikidata_id"]).group(0)
    if rec.get("id"):
        m = _QID_RE.search(rec["id"])
        if m:
            return m.group(0)
    raise ValueError(f"No Wikidata id found in record with keys {list(rec)}")

def _extract_prediction(rec: dict) -> str:
    if isinstance(rec.get("prediction"), str):
        return rec["prediction"]

    if isinstance(rec.get("translation"), str):
        return rec["translation"]

    tgt = rec.get("targets")
    if isinstance(tgt, str):
        return tgt

    if isinstance(tgt, list) and tgt:
        first = tgt[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            # try typical keys in order
            for k in ("translation", "mention", "surface", "name"):
                if k in first and isinstance(first[k], str):
                    return first[k]

    raise ValueError("Could not locate a prediction string in record "
                     f"(keys={list(rec)})")

# file loaders
def load_references(path: str, entity_filter: List[str]) -> List[dict]:
    refs: List[dict] = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            rec = json.loads(line.strip())
            if not rec.get("targets"):
                print(f"Empty target list for {rec.get('wikidata_id') or rec.get('id')}")
                continue
            if entity_filter and not any(e in rec.get("entity_types", []) for e in entity_filter):
                continue
            refs.append(rec)
    return refs

# def get_mentions(refs: List[dict]) -> Dict[str, Set[str]]:
#     mentions: Dict[str, Set[str]] = {}
#     for rec in refs:
#         iid = rec["id"]
#         mentions[iid] = {t["mention"] for t in rec["targets"] if "mention" in t}
#     return mentions
def build_mention_pool(refs: List[dict]) -> Dict[str, Set[str]]:
    pool: Dict[str, Set[str]] = {}
    for rec in refs:
        qid = _instance_id(rec)
        if qid not in pool:
            pool[qid] = set()
        for t in rec.get("targets", []):
            if "mention" in t:
                pool[qid].add(t["mention"])
    return pool

def load_predictions(path: str) -> Dict[str, str]:
    preds: Dict[str, str] = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            rec = json.loads(line.strip())
            iid = rec["id"]
            pred = _extract_prediction(rec)
            # keep first prediction if duplicates appear
            preds.setdefault(iid, pred)
    return preds

def entity_name_accuracy(preds: Dict[str, str],
                         refs: List[dict],
                         mention_pool: Dict[str, Set[str]],
                         verbose: bool = False) -> Dict[str, float]:
    correct = total = 0
    for rec in refs:
        iid = rec["id"]
        qid = _instance_id(rec)

        pred = preds.get(iid)
        mentions = mention_pool.get(qid, set())

        total += 1
        if pred is None:
            if verbose:
                print(f"No prediction for {iid} ({qid})")
            continue

        norm_pred = pred.casefold()
        if any(m.casefold() in norm_pred for m in mentions):
            correct += 1

        elif verbose:
            print(f"{iid} ({qid})\n  pred: {pred}\n  gold: {mentions}\n")

    return {"correct": correct, "total": total,
            "accuracy": correct / total if total else 0.0}

if __name__ == "__main__":
    print(f"[1/4] Loading references from {PATH_TO_REFERENCES} …")
    references = load_references(PATH_TO_REFERENCES, ENTITY_TYPES)
    gold_mentions = build_mention_pool(references)
    print(f"      {len(references)} instances kept.")

    print(f"[2/4] Loading system predictions from {PATH_TO_PREDICTIONS} …")
    predictions = load_predictions(PATH_TO_PREDICTIONS)
    print(f"      {len(predictions)} predictions loaded.")

    missing = set(gold_mentions) - set(predictions)
    if missing:
        print(f"WARNING: no prediction for {len(missing)} of "
              f"{len(gold_mentions)} reference instances.")

    print("[3/4] Scoring …")
    scores = entity_name_accuracy(predictions, references, gold_mentions, VERBOSE)

    print("\n[4/4] Results")
    print("=============================================")
    print(f"Evaluation results in {TARGET_LANGUAGE}")
    print(f"Correct instances   = {scores['correct']}")
    print(f"Total instances     = {scores['total']}")

    print("-----------------------------")
    print(f"m‑ETA               : {scores['accuracy']*100:5.2f}")
    print("=============================================")
    print("")

    print("Evaluation completed.")