from transformers import AutoTokenizer
from simpletransformers.t5 import T5Model
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import torch
import datetime
from wandb.integration.prodigy.prodigy import named_entity


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

# def annotate_entities(source, mapping):
#     found = []
#     for entity, translation in mapping.items():
#         if re.search(r'\b' + re.escape(entity) + r'\b', source):
#             found.append(f"{entity} -> {translation}")
#     return found
def load_validation_dataset(jsonl_path, mapping_file): # never used
    entity_mapping = load_entity_mapping(mapping_file)
    data = []
    with open(jsonl_path, 'r', encoding="utf-8") as file:
        for l in file:
            entry = json.loads(l)
            source = entry.get('source')
            if not source:
                continue

            target_list = entry.get('targets')
            if not target_list:
                continue

            entity_type = entry['entity_types']
            wikidata_id = entry['wikidata_id']

            for tr in target_list:
                if not tr.get('translation'):
                    continue
                target_translation = tr.get('translation', '').strip()

                entity = entity_mapping.get(wikidata_id)
                hints = f"Entities: {entity[0]} -> {entity[1]} | " if entity else ""

                entity_aware_input = f"Translate English to french Source: {source}"
                target = f"{hints}{target_translation}"
                data.append({
                    "input_text": entity_aware_input,
                    "target_text": target
                })
    print(data[0:5])
    return pd.DataFrame(data)

def load_train_dataset(jsonl_path, mapping_file):
    entity_mapping = load_entity_mapping(mapping_file)
    data = []
    with open(jsonl_path, 'r', encoding="utf-8") as file:
        for l in file:
            entry = json.loads(l)
            source = entry.get('source')
            if not source:
                continue

            target = entry.get('target')
            if not target:
                continue

            entities = entry['entities']
            hints = f"Entities: "
            for id in entities:
                entity = entity_mapping.get(id)
                hints = hints + f"{entity[0]} -> {entity[1]} | " if entity else ""

            entity_aware_input = f"Translate English to French Source: {source}"
            t = f"{hints}{target}"
            data.append({
                "input_text": entity_aware_input,
                "target_text": t
            })
    print(data[0:5])
    return pd.DataFrame(data)

if __name__ == "__main__":
    model_args = {
        "num_beams": 5,
        "overwrite_output_dir": True,
        "max_seq_length": 128,
        "early_stopping": True,
        "early_stopping_patience": 3,
        "train_batch_size": 16,
        "eval_batch_size": 16,
        "num_train_epochs": 15,
        "learning_rate": 1e-3,
        "save_steps": -1,
        "save_eval_checkpoints": False,
        "save_model_every_epoch": False,
        "evaluate_during_training": True,
        "use_cuda": torch.cuda.is_available(),
        "fp16": False,
        "wandb_entity": "https://wandb.ai/itsdianamxn-universitatea-alexandru-ioan-cuza-din-ia-i",
        "wandb_project": "ea-mt_semeval",
        "wandb_kwargs": {"name": f"mt5_small_run_{datetime.datetime.now().strftime("%m.%d_%H")}"}
    }
    df = load_train_dataset("C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/data/semeval/train/fr/train.jsonl",
                                "C:/Users/diana/Desktop/Anul III/Licenta/SemEval/wikidata_labels_en_fr.jsonl")

    train_df, eval_df = train_test_split(df, test_size=0.1, random_state=42)
    train_df["prefix"] = ""
    eval_df["prefix"] = ""
    train_df = train_df[["prefix", "input_text", "target_text"]]
    eval_df = eval_df[["prefix", "input_text", "target_text"]]

    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small", legacy=False)
    # entities = set()
    # with open("wikidata_labels_en_fr.jsonl", 'r', encoding="utf-8") as f:
    #     for line in f:
    #         entry = json.loads(line)
    #         entities.add(entry["source"])
    #         entities.add(entry["target"])
    #
    # tokenizer.add_tokens(list(entities))

    model = T5Model("mt5", "google/mt5-small", args=model_args, tokenizer=tokenizer)
    model.model.resize_token_embeddings(len(tokenizer))

    model.train_model(train_df, eval_data=eval_df)

