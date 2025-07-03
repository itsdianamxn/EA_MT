#### mention sentence in input; mention sentence in output ####

from transformers import AutoTokenizer, MT5Model
from simpletransformers.t5 import T5Model
import pandas as pd
import json
import torch
import datetime
import os

crt_model = "mt5"

languages = {#"ar": "Arabic",
              "fr": "French",
            #  "it": "Italian",
             # "de": "German",
            #  "es": "Spanish",
            #  "ja": "Japanese",
            #  "ko": "Korean"
             }

# PROJECT_ROOT = "/content/drive/MyDrive/"
PROJECT_ROOT = "../"

def load_entity_mapping(file_path):
    mapping = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for l in f:
            entry = json.loads(l)
            id = entry.get("id", "").strip()
            source = entry.get("source", "").strip()
            target = entry.get("target", "").strip()
            mapping[id] = [source, target]
    return mapping


def build_entity_and_sentence_frames(jsonl_path, wikidata_map, prefix):
    ent_rows, sent_rows = [], []
    i = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            src = rec.get("source", "").strip()
            tgt = rec.get("target", "").strip()
            eng_ents = []
            trans_ents = []
            for qid in rec.get("entities", []):
                labels = wikidata_map.get(qid)
                if labels:
                    if i% 2 == 0:
                        ent_rows.append({"input_text": f'{labels[0]}., "entities": [{labels[0]}]',  # English
                                         "target_text": f'{labels[1]}., "entities": [{labels[1]}]',  # Translated
                                         "prefix": prefix})
                    else:
                        ent_rows.append({"input_text": f'{labels[0]}, "entities":[{labels[0]}]',  # English
                                         "target_text": f'{labels[1]}, "entities":[{labels[1]}]',  # Translated
                                         "prefix": prefix})
                    eng_ents.append(labels[0])
                    trans_ents.append(labels[1])
            sent_rows.append({"input_text": f'{src}, "entities":[{", ".join(eng_ents)}]',
                              "target_text": f'{tgt}, "entities":[{", ".join(trans_ents)}]',
                              "prefix": prefix})
            i += 1

    print("Entity rows:", ent_rows[0:3])
    print("Sentence rows:", sent_rows[0:3])
    return ent_rows, sent_rows

def load_validation(validation_path, entity_map, prefix):
    rows = []
    with open(validation_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            src = rec.get("source", "").strip()
            wid = rec.get("wikidata_id", "").strip()
            for t in rec.get("targets", []):
                tgt = t.get("translation", "").strip()
                if entity_map.get(wid):
                    rows.append({
                        "input_text": f'{src}, "entities":[{entity_map[wid][0]}]',
                        "target_text": f'{tgt}, "entities":[{entity_map[wid][1]}]',
                        "prefix": prefix
                    })
                else:
                    rows.append({
                        "input_text": src,
                        "target_text": tgt,
                        "prefix": prefix
                    })
    print("Validation rows:", rows[0:3])
    return rows


def add_language(trains, validations, lang_id, lang_name):
    print(f"Processing {lang_name} ({lang_id})...")
    if not os.path.isfile(f"{PROJECT_ROOT}/Train/{lang_id}/train.jsonl") or \
            not os.path.isfile(f"{PROJECT_ROOT}/Wikidata/wikidata_labels_en_{lang_id}.jsonl") or \
            not os.path.isfile(f"{PROJECT_ROOT}/Validation/{lang_id}.jsonl"):
        print(f"Skipping {lang_name} ({lang_id}) due to missing files.")
        return

    entity_map = load_entity_mapping(f"{PROJECT_ROOT}/Wikidata/wikidata_labels_en_{lang_id}.jsonl")
    entities, sentences = build_entity_and_sentence_frames(f"{PROJECT_ROOT}/Train/{lang_id}/train.jsonl",
                                                           entity_map,
                                                           f"translate English to {lang_name}")
    trains.extend(entities)
    trains.extend(sentences)

    validations.extend(
        load_validation(f"{PROJECT_ROOT}/Validation/{lang_id}.jsonl",
                        entity_map,
                        f"translate English to {lang_name}"))


model_args = {
      "num_beams": 5,
      "overwrite_output_dir": True,
      "max_seq_length": 64,
      "early_stopping": True,
      "early_stopping_patience": 3,
      "train_batch_size":8,
      "gradient_accumulation_steps": 2,
      "eval_batch_size": 8,
      "num_train_epochs": 4,
      "dropout_rate": 0.1,
      "scheduler":"cosine_schedule_with_warmup",
      "encoding":"utf-8",
      "warmup_steps":1,
      "save_eval_checkpoints": False,
      "save_model_every_epoch": True,
      "evaluate_during_training": True,
      "use_cuda": torch.cuda.is_available(),
      "fp16": False,
      "output_dir": "/content/drive/MyDrive/DE_enti_out_1",
      "wandb_entity": "https://wandb.ai/itsdianamxn-universitatea-alexandru-ioan-cuza-din-ia-i",
      "wandb_project": "ea-mt_semeval",
      "wandb_kwargs": {"name": f"{crt_model}_wiki_fr_{datetime.datetime.now().strftime('%m.%d_%H')}"}
}
if __name__ == "__main__":

    trains = []
    validations = []
    for lId in languages:
        add_language(trains, validations, lId, languages[lId])
    df_train = pd.DataFrame(trains)
    df_valid = pd.DataFrame(validations)

    tokenizer = AutoTokenizer.from_pretrained(f"google/{crt_model}-base", use_fast=False, legacy=False)
    model = T5Model(crt_model, f"google/{crt_model}-base", args=model_args, tokenizer=tokenizer)
    model.train_model(df_train, eval_data=df_valid)