import os
import json
import torch
from transformers import AutoTokenizer, MT5Model
from simpletransformers.t5 import T5Model

from CurriculumTraining.generalised_mt5 import LANGUAGECODE, LANGUAGENAME

#PROJECT_ROOT = "/content/drive/MyDrive/"
PROJECT_ROOT = "."
LANGUAGECODE = "ro"
LANGUAGENAME = "Romanian"
MODEL = "t5"
def build_inputs(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            source = entry.get("source", "").strip()
            input_text = f"translate English to {LANGUAGENAME} and extract the entities: {source}"
            data.append(input_text)
    print(data[0:2])
    return data

def run():
    model_path = f"{PROJECT_ROOT}/Models/{LANGUAGECODE}_enti_out_t5"
    input_path = f"{PROJECT_ROOT}/Test/fr_FR.jsonl"
    output_path = f"{PROJECT_ROOT}/Results/results_{LANGUAGECODE}_{MODEL}_enti_out.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            legacy=False)

    model_args = {
        "use_cuda": torch.cuda.is_available(),
        "max_length": 128,
        "eval_batch_size": 32,
        "num_beams": 3,
        "overwrite_output_dir": True,
        "save_eval_checkpoints": False,
        "fp16": False
    }
    model = T5Model(MODEL, model_path, tokenizer=tokenizer, args=model_args)

    batch_inputs = build_inputs(input_path)

    print(f"Predicting {len(batch_inputs)} examples...")
    predictions = model.predict(batch_inputs)

    templates = []
    with open(input_path, 'r', encoding='utf-8') as templates_file:
        for line in templates_file:
            templates.append(json.loads(line))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for entry, prediction in zip(templates, predictions):
            entry["targets"] = [prediction]
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write('\n')

    print(f"Predictions saved in {output_path}")

if __name__ == '__main__':
    run()