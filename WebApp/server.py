import json
import unicodedata
import torch
from simpletransformers.t5 import T5Model
from transformers import AutoTokenizer
from flask import Flask, jsonify, request, Response
from google import genai
import deepl
client = genai.Client()
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
    return f"translate English to French Source: {source}"

def load_model():
    #model_path = "C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/Scripts/CurriculumTraining/outputs/best_model"
    model_path = "C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/Models/Curriculum_Training/best_model"
    tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            legacy=False)

    model_args = {
        "use_cuda": torch.cuda.is_available(),
        "max_length": 128,
        "eval_batch_size": 16,
        "num_beams": 3,
        "overwrite_output_dir": True,
        "save_eval_checkpoints": False,
        "fp16": False
    }
    print("CUDA available:", torch.cuda.is_available())
    print("Model args:", model_args)

    model = T5Model("t5", model_path, tokenizer=tokenizer, args=model_args)
    model.model.resize_token_embeddings(len(tokenizer))

    return model
    #entity_mapping = [] #load_entity_mapping("C:/Users/diana/Desktop/Anul III/Licenta/SemEval/wikidata_labels_en_fr.jsonl")

app = Flask(__name__)

@app.route('/webapp/<path:path>', methods=['GET'])
def webapp(path):
    print(request.path)

    full_path = "C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/WebApp/" + path
    with open(full_path, 'rb') as f:
        content = f.read()
        if path.endswith(".css"):
            return Response(content, mimetype="text/css")
        if path.endswith(".js"):
            return Response(content, mimetype="application/javascript")
        return Response(content)

@app.route('/translate', methods=['POST'])
def translate():
    body = json.loads(request.data.decode('utf-8'))
    print("Decoded request body:", body)
    sources = body.get('sources')
    engine = body.get('engine')
    source_lang = body.get('source_lang')
    target_lang = body.get('target_lang')
    print("Incoming request to translate phrases:", sources)
    translations = []
    for source in sources:
        source = source.replace(" a play", " a theatrical piece")
        input = f"translate {source_lang} to {target_lang}: {source}"
        translations.append(input)
    print("Input for model:", translations)
    if engine == "Deepl":
        auth_key = "9e2244de-6236-45fc-afc8-a6a8e55b473b:fx"
        translator = deepl.Translator(auth_key)

        prediction = translator.translate_text("\n".join(sources), target_lang=target_lang[0:2].upper(), source_lang=source_lang[0:2].upper())
        prediction = prediction.text.split('\n')
    elif engine == "ChatGPT":
        prompt = f"You are an expert translator. Translate from {source_lang} to {target_lang}.\
Only provide the translation without explanations. {"\n".join(sources)}"
        print ("Prompt for gemini:", prompt)
        response = client.models.generate_content(  
            model="gemini-2.0-flash-001",
            contents=prompt,
        )
        print(response)
        prediction = response.text.split('\n')
    elif engine == "Experimental":
        prediction = model.predict(translations)


    print("Responding with prediction:", prediction)
    return jsonify({'source_lang': source_lang, 'target_lang': target_lang, 'source': source, 'translated' : prediction})
# @app.route('/translate', methods=['GET'])
# def translate():
#     source = unquote(request.args.get('phrase'))
#     engine = request.args.get('engine')
#     source_lang = request.args.get('source_lang')
#     target_lang = request.args.get('target_lang')
#     print("Incoming request to translate phrase:", source)
#     input = f"translate {source_lang} to {target_lang}: {source}"
#     print("Input for model:", input)
#     if engine == "Deepl":
#
#         auth_key = "9e2244de-6236-45fc-afc8-a6a8e55b473b:fx"
#         translator = deepl.Translator(auth_key)
#
#         prediction = translator.translate_text(source, target_lang=target_lang[0:2].upper(), source_lang=source_lang[0:2].upper())
#         prediction = prediction.text
#     elif engine == "chatGPT":
#         # Call Google API
#         pass
#     elif engine == "Experimental":
#         prediction = model.predict([input])
#
#
#     print("Responding with prediction:", prediction)
#     return jsonify({'source_lang': source_lang, 'target_lang': target_lang, 'source': source, 'translated' : prediction})

if __name__ == '__main__':
    print("Loading model...")
    model = load_model()
    print("Model loaded. Starting server...")

    app.run()