import json
import os
import torch
from transformers import AutoTokenizer
from simpletransformers.t5 import T5Model

def load_entity_mapping(path):
    mapping = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            mapping[entry["id"]] = [entry["source"], entry["target"]]
    return mapping

def build_inputs_with_entities(input_path, entity_mapping):
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            source = entry.get("source", "")
            entities = entry.get("entities", [])

            hints = []
            for wikidata_id in entities:
                if wikidata_id in entity_mapping:
                    en, fr = entity_mapping[wikidata_id]
                    hints.append(f'"{en}": "{fr}"')

            hint_str = f'Entities: {{{", ".join(hints)}}} | ' if hints else ""
            input_text = f"translate English to French: {hint_str}Source: {source}"

            data.append({
                "id": entry["id"],
                "input_text": input_text,
                "source": source
            })
    print(data[0:2])
    return data

def predict_and_save(model_path, input_file, mapping_file, output_file):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model_args = {
        "use_cuda": torch.cuda.is_available(),
        "eval_batch_size": 16,
        "num_beams": 5,
        "fp16": False
    }

    model = T5Model("mt5", model_path, tokenizer=tokenizer, args=model_args)

    entity_mapping = load_entity_mapping(mapping_file)
    inputs = build_inputs_with_entities(input_file, entity_mapping)
    predictions = model.predict([x["input_text"] for x in inputs])

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry, prediction in zip(inputs, predictions):
            output = {
                "id": entry["id"],
                "source_language": "English",
                "target_language": "French",
                "text": entry["source"],
                "prediction": prediction
            }
            json.dump(output, f, ensure_ascii=False)
            f.write('\n')

    print(f"Saved predictions to {output_file}")

if __name__ == "__main__":
    predict_and_save(
        model_path="C:/Users/diana/Desktop/Anul III/Licenta/SemEval/trained_with_50_epochs(setdedatemaimare)",
        input_file="C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/data/references/validation/FR_fr.jsonl",
        mapping_file="C:/Users/diana/Desktop/Anul III/Licenta/SemEval/wikidata_labels_en_fr.jsonl",
        output_file="C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/data/predictions/trained_mt5_model/validation/FR_fr.jsonl"
    )
