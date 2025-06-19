from transformers import AutoTokenizer
from simpletransformers.t5 import T5Model
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import torch
import datetime


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

def load_train_dataset(jsonl_path, mapping_file): # never used
    entity_mapping = load_entity_mapping(mapping_file)
    data = []
    with open(jsonl_path, 'r', encoding="utf-8") as file:
        for l in file:
            entry = json.loads(l)
            source = entry.get('source')
            if not source:
                continue

            target_translation = entry.get('target')
            if not target_translation:
                continue

            entities = entry['entities']
            hints = None
            for wikidata_id in entities:
                entity = entity_mapping.get(wikidata_id)
                if not entity:
                    continue
                hints = (hints + ', ' if hints else '') + f'"{entity[0]}": "{entity[1]}"'
            hints = f"Entities: {{{hints}}} |" if hints else ""

            entity_aware_input = f"translate English to French: {hints} Source: {source}"
            target = target_translation
            data.append({
                "input_text": entity_aware_input,
                "target_text": target
            })
    print(data[7:8])
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
        "wandb_kwargs": {"name": f"mt5_small_wiki_{datetime.datetime.now().strftime('%m.%d_%H')}"}
    }
    df = load_train_dataset("C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/data/semeval/train/fr/train.jsonl",
                            "C:/Users/diana/Desktop/Anul III/Licenta/SemEval/wikidata_labels_en_fr.jsonl")

    train_df, eval_df = train_test_split(df, test_size=0.1, random_state=42)
    train_df["prefix"] = ""
    eval_df["prefix"] = ""
    train_df = train_df[["prefix", "input_text", "target_text"]]
    eval_df = eval_df[["prefix", "input_text", "target_text"]]

    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small", legacy=False)

    #model = T5Model("mt5", "google/mt5-small", args=model_args, tokenizer=tokenizer)
    #model.model.resize_token_embeddings(len(tokenizer))

    # model.train_model(train_df, eval_data=eval_df)