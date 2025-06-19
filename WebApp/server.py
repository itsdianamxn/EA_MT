import json
import unicodedata
import torch
from simpletransformers.t5 import T5Model
from transformers import AutoTokenizer
from flask import Flask, jsonify, request, Response
from google import genai
import deepl

client = genai.Client()
languages = {"Arabic": "ar",
             "French": "fr",
             "Italian": "it",
             "German": "de",
             "Spanish": "es",
             "Japanese": "ja",
             "Romanian": "ro",
             }
crt_lang_ea = ""
ea_model = None

model_args = {
    "use_cuda": torch.cuda.is_available(),
    "max_length": 128,
    "eval_batch_size": 8,
    "num_beams": 3,
    "fp16": False
}

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

def load_model():
#    model_path = "C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/Scripts/CurriculumTraining/outputs/best_model"
    model_path = f"C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/Scripts/Models/epoch_4+1_ALL"
#    model_path = f"C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/Scripts/Models/french_t5/Base_5_epoch"

    tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            legacy=False)


    print("CUDA available:", torch.cuda.is_available())
    print("Model args:", model_args)
    model = T5Model("mt5", model_path, tokenizer=tokenizer, args=model_args)
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
    lang_id = languages[target_lang]
    entities = []
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
    elif engine == "Gemini":
        prompt = (f"You are an expert translator. Translate from {source_lang} to {target_lang}. " +
                f"Generate the translation for the sentence without any additional explanations and without the {source_lang} text, " +
                "ensuring that named entities are translated accurately, " + 
                "then provide a list with the named entities in the translation.\n"+
                'Example:\nFor the phrase: "How many Academy Awards did Silence of the Lambs win?" ' +
                'the output will be: Combien d’Oscars du cinéma le film Le Silence des agneaux a-t-il remportés ?\n' +
                f'Entities: Oscars du cinéma, Le Silence des agneaux\n{sources}')
        print ("Prompt for gemini:", prompt)
        response = client.models.generate_content(  
            model="gemini-2.0-flash-001",
            contents=prompt,
        )
        out = response.text.split('\n')
        print("gemini response:", out)
        prediction = []
        prediction.append(out[0])
        idx = out[1].find('Entities: ')
        if idx != -1:
            entities_str = out[1][idx + len('Entities: '):]
            entities_list = entities_str.split(', ')
            entities.append(entities_list)
                
    elif engine == "Entity-Aware":
        update_ea_model(lang_id)
        ea_prediction = ea_model.predict(translations)
        print("Entity-aware prediction:", ea_prediction)
        prediction = []
        for pred in ea_prediction:
            idx = pred.find(', "entities":[')
            if idx != -1:
                entities_str = pred[idx + 1 + len(', "entities":['):-2]
                if entities_str != "":
                    print("Entities string:", entities_str)
                    entities_list = entities_str.split('", "')
                    print("Entities list:", entities_list)
                    entities.append(entities_list)
            prediction.append(pred[:idx])

    elif engine == "Experimental":
        prediction = model.predict(translations)


    print("Responding with prediction:", prediction)
    return jsonify({'source_lang': source_lang, 'target_lang': target_lang, 'source': source, 'translated' : prediction, 'entities': entities})

def update_ea_model(target_lang):
    global crt_lang_ea, ea_model
    if crt_lang_ea != target_lang:
        if target_lang in ["fr", "it", "ro", "de"]:
            ea_model_path = f"C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/Scripts/Models/{target_lang}_enti_out_t5"
            model = "t5"

        else:
            ea_model_path = f"C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/Scripts/Models/{target_lang}_enti_out"
            model = "mt5"

        tokenizer = AutoTokenizer.from_pretrained(
                ea_model_path,
                use_fast=False,
                legacy=False)

        ea_model = T5Model(model, ea_model_path, tokenizer=tokenizer, args=model_args)

        crt_lang_ea = target_lang

if __name__ == '__main__':
    print("Loading model...")
    model = load_model()
    print("Model loaded. Starting server...")
    
    app.run()