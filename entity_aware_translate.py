import os
import json
import torch
from simpletransformers.t5 import T5Model
from transformers import AutoTokenizer
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

def load_test_examples(file_path):
    data = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                data.append(entry)
    return data

def build_entity_aware_input(entry, entity_mapping):
    source = entry.get("source", "").strip()
    entity_type = entry['entity_types']
    wikidata_id = entry.get('wikidata_id')
    entity = entity_mapping.get(wikidata_id)
    hints = f"Entities: {entity[0]} -> {entity[1]} |" if entity else ""
    #
    # if entry.get("wikidata_id") or entry.get("entity_types"):
    #     mention = entry.get("source", "").strip()
    #     parts = [mention]
    #
    #     if entry.get("entity_types"):
    #         parts.append(f"type: {', '.join(entry['entity_types'])}")
    #     if entry.get("wikidata_id"):
    #         parts.append(f"WD: {entry['wikidata_id']}")

    #    entity_info.append(f"{mention} ({', '.join(parts[1:])})" if len(parts) > 1 else mention)
    return f"translate English to French: {hints} Source: {source}"

def run_model():
    model_path = "C:/Users/diana/Desktop/Anul III/Licenta/SemEval/entity_aware_15_epochs/best_model"
    input_path = "C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/data/references/test/fr_FR.jsonl"
    output_path = "C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/data/predictions/trained_mt5_model/test1/FR_fr.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    # Add entities as special tokens
    # entities = set()
    # with open("C:/Users/diana/Desktop/Anul III/Licenta/SemEval/wikidata_labels_en_fr.jsonl", 'r',
    #           encoding='utf-8') as f:
    #     for line in f:
    #         entry = json.loads(line)
    #         entities.add(entry['source'])
    #         entities.add(entry['target'])
    #
    # tokenizer.add_tokens(list(entities))

    model_args = {
        "use_cuda": torch.cuda.is_available(),
        "eval_batch_size": 16,
        "num_beams": 3,
        "overwrite_output_dir": True,
        "save_eval_checkpoints": False,
        "no_cache": True,
        "fp16": True
    }

    model = T5Model("mt5", model_path, tokenizer=tokenizer, args=model_args)
    model.model.resize_token_embeddings(len(tokenizer))

    entity_mapping = load_entity_mapping("C:/Users/diana/Desktop/Anul III/Licenta/SemEval/wikidata_labels_en_fr.jsonl")
    test_data = load_test_examples(input_path)
    batch_inputs = [build_entity_aware_input(entry, entity_mapping) for entry in test_data]

    print(f"Predicting {len(batch_inputs)} examples...")
    predictions = model.predict(batch_inputs)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for entry, prediction in zip(test_data, predictions):
            entry["targets"] = prediction
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write('\n')

    print(f"Predictions saved in {output_path}")


if __name__ == '__main__':
    print("CUDA available:", torch.cuda.is_available())
    run_model()
