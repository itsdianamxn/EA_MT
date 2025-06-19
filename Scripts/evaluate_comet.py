import json
import os

from comet import download_model, load_from_checkpoint
import torch

COMET_MODEL_NAME = "Unbabel/wmt22-comet-da"
BATCH_SIZE = 8        # smaller batch helps on CPU

SYSTEM_NAME = "trained_mt5_model"
SOURCE_LANGUAGE = "en_US"
TARGET_LANGUAGE = "es_ES"
DATA_DIR = "../data"
SPLIT = "test"
NUM_GPUS = 1 if torch.cuda.is_available() else 0
BATCH_SIZE = 32

# PATH_TO_REFERENCES = os.path.join(
#     "C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/data/predictions_for_comparison/fr_FR.jsonl"
# )
PATH_TO_REFERENCES = "C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/data/predictions_for_comparison/it_IT.jsonl"
#PATH_TO_REFERENCES = "C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/data/predictions_for_comparison/de_DE.jsonl"

# The path to the predictions is formatted as follows:
# data/predictions/{system_name}/{split}/{target_language}.jsonl
# PATH_TO_PREDICTIONS = os.path.join(
#     #"C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/data/predictions/trained_mt5_model/test/fr_FR.jsonl", #antrenate cu format Source: "source" si Target: Entities: "entities" | "target"

#     #"C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/data/predictions/curriculum_training/t5_extended_dataset/results_base_extended.jsonl"
#     #"C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/data/predictions/curriculum_training/test/fr_FR_1.jsonl"
#     #"C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/data/predictions/trained_t5base_model/results_base.jsonl" #e antrenat cu t5
#     #"C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/data/predictions/trained_t5base_model/fr_base_1_epoch_wiki.jsonl" #antrenat cu t5 si cu entitate si traducere in prompt translate English to French: Entities: {"{en}": "{fr}", ...} Source: {source}
#     "C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/data/predictions/trained_t5base_model/results_base_5ep_lr=1e-4.jsonl"
#     #"C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/data/predictions/trained_t5base_model/FR_fr_wiki_5ep.jsonl" #antrenat cu t5 si cu entitate si traducere in prompt translate English to French: Entities: {"{en}": "{fr}", ...} Source: {source}
#     #"C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/data/predictions/trained_t5base_model/results_it.jsonl"
#
# )
#PATH_TO_PREDICTIONS = "C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/data/predictions/curriculum_training/t5_extended_dataset/rbe_it_new.jsonl"
#PATH_TO_PREDICTIONS = "C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/data/predictions/trained_mt5_model/double_extended_base/results_it.jsonl"
PATH_TO_PREDICTIONS = "C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/Scripts/Results/results_it_t5_t5_enti_out.jsonl"

#PATH_TO_PREDICTIONS = "C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/data/predictions/trained_mt5_model/extended/results_de.jsonl"
references = {}

with open(PATH_TO_REFERENCES, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        references[data["id"]] = data

print(f"Loaded {len(references)} references from {PATH_TO_REFERENCES}")

# Load the predictions
predictions = {}

with open(PATH_TO_PREDICTIONS, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        if idx == 0:
            print(line.strip())
        data = json.loads(line)
        predictions[data["id"]] = data

print(f"Loaded {len(predictions)} predictions from {PATH_TO_PREDICTIONS}")

# Get all those references that have a corresponding prediction
ids = set(references.keys()) & set(predictions.keys())
num_missing_predictions = len(references) - len(ids)

if num_missing_predictions > 0:
    print(f"Missing predictions for {num_missing_predictions} references")
else:
    print("All references have a corresponding prediction")
instance_ids = {}
instances = []
current_idx = 0

for _id in sorted(ids):
    ref_obj = references[_id]
    hyp_obj = predictions[_id]
    instances.append(
        {
            "src": ref_obj["source"],  # English sentence
            "ref": ref_obj["targets"][0]["translation"],  # gold translation
            "mt": hyp_obj["targets"][0]
        }
    )

    instance_ids[_id] = (current_idx, current_idx + 1)
    current_idx += 1

print(f"Created {len(instances)} instances")

model_path = download_model(COMET_MODEL_NAME)

model = load_from_checkpoint(model_path)

outputs = model.predict(instances, batch_size=BATCH_SIZE, gpus=NUM_GPUS)

scores = outputs.scores
max_scores = []

for id, indices in instance_ids.items():
    max_score = max(scores[indices[0]: indices[1]])
    max_scores.append(max_score)

system_score = sum(max_scores) / (len(max_scores) + num_missing_predictions)

print(f"Average COMET score: {100. * system_score:.2f}")

print(outputs.system_score)