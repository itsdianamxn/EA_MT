import json
import pandas as pd
from sklearn.model_selection import train_test_split
from simpletransformers.t5 import T5Model
from transformers import AutoTokenizer
import os
import torch


def load_data(jsonl_path):
    data = []
    with open(jsonl_path, 'r', encoding="utf-8") as file:
        for line in file:
            entry = json.loads(line)
            source_text = entry.get('source', "").strip()
            target_text = str(entry.get('target', "")).strip()
            if source_text and target_text:
                data.append({
                    "input_text": source_text,
                    "target_text": target_text
                })
    return pd.DataFrame(data)

if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())

    #file_path = input("Introduceți calea către fișierul de date (.jsonl): ")
    df = load_data("C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/Data/semeval/semeval/train/fr/train.jsonl")
    print(f"Numărul de eșantioane încărcate: {len(df)}")

    if not df.empty:
        print("Primul rând din DataFrame:")
        print(df.iloc[0])

    df["prefix"] = "translate English to French: "

    df.to_csv('french_translation_data.csv', index=False)

    train_df, eval_df = train_test_split(df, test_size=0.1, random_state=42)
    print(f"Training dataset size: {len(train_df)}")
    print(f"Evaluation dataset size: {len(eval_df)}")

    # model_args = {
    #     "num_beams": 5,
    #     "overwrite_output_dir": True,
    #     "max_seq_length": 64,
    #     "train_batch_size": 16,
    #     "eval_batch_size": 16,
    #     "num_train_epochs": 64,
    #     "learning_rate": 5e-4,
    #     "save_steps": -1,
    #     "save_eval_checkpoints": False,
    #     "save_model_every_epoch": False,
    #     "evaluate_during_training": True,
    #     "use_cuda": torch.cuda.is_available(),
    #     "fp16": True,
    # }
    model_args = {
        "num_beams": 5,
        "overwrite_output_dir": True,
        "max_seq_length": 128,
        "train_batch_size": 16,
        "eval_batch_size": 16,
        "num_train_epochs": 128,
        "learning_rate": 3e-4,
        "save_steps": -1,
        "save_eval_checkpoints": False,
        "save_model_every_epoch": False,
        "evaluate_during_training": True,
        "use_cuda": True,
        "fp16": False,
        "warmup_steps": 100,
        "wandb_entity": "https://wandb.ai/itsdianamxn-universitatea-alexandru-ioan-cuza-din-ia-i",
        "wandb_project": "ea-mt_semeval",  # your wandb project name
        "wandb_kwargs": {"name": "mt5_small_run_3"}  # additional wandb parameters if needed
    }

    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small", legacy=False)
    model = T5Model("mt5", "google/mt5-small", args=model_args, tokenizer=tokenizer, use_fast=True)

    print("Starting training...")
    model.train_model(train_df, eval_data=eval_df)
    print("Training completed.")

    print("Running evaluation...")
    results = model.eval_model(eval_df)
    print(f"Evaluation results: {results}")

    print("Making predictions...")
    sample_texts = [
        "translate English to French: What is the capital of France?",
        "translate English to French: Who wrote the book 1984?"
    ]
    predictions = model.predict(sample_texts)
    with open("out.txt", 'w', encoding='utf-8') as f:
        for text, prediction in zip(sample_texts, predictions):
            f.write(f"Input: {text}\nOutput: {prediction}\n")