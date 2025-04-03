# General imports
import json
import os
import torch.serialization
from comet.models.utils import Prediction
# Import the comet module for the evaluation
from comet import download_model, load_from_checkpoint
COMET_MODEL_NAME = "Unbabel/wmt22-comet-da"
SYSTEM_NAME = "trained_mt5_model"
SOURCE_LANGUAGE = "en_US"
TARGET_LANGUAGE = "fr_FR"
DATA_DIR = "../data"
SPLIT = "validation"
NUM_GPUS = 1
BATCH_SIZE = 32

# The path to the references is formatted as follows:
# data/references/{split}/{target_language}.jsonl
PATH_TO_REFERENCES = os.path.join(
    "C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/data/references/validation",
    f"{TARGET_LANGUAGE}.jsonl",
)

# The path to the predictions is formatted as follows:
# data/predictions/{system_name}/{split}/{target_language}.jsonl
PATH_TO_PREDICTIONS = os.path.join(
    "C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/data/predictions/trained_mt5_model/validation",
    f"{TARGET_LANGUAGE}.jsonl",
)

references = {}

with open(PATH_TO_REFERENCES, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        references[data["id"]] = data

print(f"Loaded {len(references)} references from {PATH_TO_REFERENCES}")

# Load the predictions
predictions = {}

with open(PATH_TO_PREDICTIONS, "r", encoding="utf-8") as f:

    for line in f:
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
    current_index = 0

    for id in sorted(list(ids)):
        reference = references[id]
        prediction = predictions[id]

        for target in reference["targets"]:
            instances.append(
                {
                    "src": reference["source"],
                    "ref": target["translation"],
                    "mt": prediction["prediction"],
                }
            )

        instance_ids[id] = [current_index, current_index + len(reference["targets"])]
        current_index += len(reference["targets"])

    print(f"Created {len(instances)} instances")

    # Download the model
    model_path = download_model(COMET_MODEL_NAME)

    # Load the model
    model = load_from_checkpoint(model_path)
    torch.serialization.add_safe_globals([Prediction])

    # Compute the scores
    outputs = model.predict(instances, batch_size=BATCH_SIZE, gpus=NUM_GPUS)

    # Extract the scores
    scores = outputs.scores
    max_scores = []

    for id, indices in instance_ids.items():
        # Get the max score for each reference
        max_score = max(scores[indices[0]: indices[1]])
        max_scores.append(max_score)

    # Compute the average score while taking into account the missing predictions (which are considered as 0)
    system_score = sum(max_scores) / (len(max_scores) + num_missing_predictions)

    print(f"Average COMET score: {100. * system_score:.2f}")

    outputs.system_score