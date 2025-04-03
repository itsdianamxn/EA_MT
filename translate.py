import os
import json
import torch
from simpletransformers.t5 import T5Model
from transformers import AutoTokenizer

def load_test_examples(file_path):
    data = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                data.append(entry)
    return data

def run_model():
    model_path = "C:/Users/diana/Desktop/Anul III/Licenta/SemEval/trained_with_128_epochs(setdedatemaimare)"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model_args = {
        "use_cuda": torch.cuda.is_available(),
        "eval_batch_size": 16,
        "num_beams": 3,
#        "max_length": 64,
        "overwrite_output_dir": True,
        "save_eval_checkpoints": False,
        "no_cache": True,
        "fp16": True
    }

    model = T5Model(
        model_type="mt5",
        model_name=model_path,
        tokenizer=tokenizer,
        args=model_args
    )

    # Load your reference/test data
    input_path = "C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/data/references/validation/FR_fr.jsonl"
    test_data = load_test_examples(input_path)

    # Prepare inputs
    batch_inputs = []
    batch_meta = []
    for entry in test_data:
        text_to_translate = entry["source"]
        batch_inputs.append(f"translate English to French: {text_to_translate}")
        # Keep a dict of metadata for each line
        batch_meta.append({
            "id": entry["id"],
            "text": text_to_translate
        })
    # If GPU runs out of memory:
    # predictions = []
    # chunk_size = 32
    # for i in range(0, len(batch_inputs), chunk_size):
    #     chunk_preds = model.predict(batch_inputs[i:i+chunk_size])
    #     predictions.extend(chunk_preds)
    predictions = model.predict(batch_inputs)

    # Write JSONL lines to output, one line per translation
    output_path = "C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/data/predictions/trained_mt5_model/validation/FR_fr.jsonl"
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for meta, pred_text in zip(batch_meta, predictions):
            output_dict = {
                "id": meta["id"],
                "source_language": "English",  # or read from the data if needed
                "target_language": "French",
                "text": meta["text"],
                "prediction": pred_text
            }
            json.dump(output_dict, outfile, ensure_ascii=False)
            outfile.write('\n')

    print(f"Predictions saved in {output_path}")

if __name__ == '__main__':
    print("CUDA available:", torch.cuda.is_available())
    run_model()
