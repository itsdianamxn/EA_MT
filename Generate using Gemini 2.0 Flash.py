import json
import secrets
import time

from google import genai
client = genai.Client()

def load_entity_mapping(file_path):
    mapping = {}
    i = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for l in f:
            i += 1
            if i < 125:
                continue
            entry = json.loads(l)
            id = entry.get("id", "").strip()
            source = entry.get("source", "").strip()
            target = entry.get("target", "").strip()
            if target == "":
                continue
            mapping[id] = [source, target]
    return mapping
def generate(sources, qid, lang_id, lang_name):

        prompt = (f"Generate a fact and its translation in {lang_name}, "+
                  f'where the named entity "{sources[0]}" is translated as: "{sources[1]}". ' +
                  f"Do not use quotation marks. The two sentences are separated by a newline character.\n"
                  )

        print ("Prompt for gemini:", prompt)
        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=prompt,
        )
        print(response)
        prediction = response.text.split('\n')

        output = f"{{\"id\": \"{secrets.token_hex(4)}\", \"source_locale\": \"en\", \"target_locale\": \"{lang_id}\", \"source\": \"{prediction[0]}\", \"target\": \"{prediction[1]}\", \"entities\": \"{qid}\", \"from\": \"generated\"}}\n"
        print(output)
        path = f"C:\\Users\\diana\\Desktop\\Anul III\\Licenta\\PROIECT\\Scripts\\Train\\{lang_id}\\train_gen.jsonl"
        with open(path, 'a', encoding='utf-8') as f:
            f.write(output)
            


if __name__ == '__main__':
    languages = {
        "ar": "Arabic",
#         "fr": "French",
#         "it": "Italian",
#         "de": "German",
#         "es": "Spanish",
#         "ja": "Japanese",
    }
    for lid in languages:
        mapping = load_entity_mapping(f'C:\\Users\\diana\\Desktop\\Anul III\\Licenta\\PROIECT\\Scripts\\Wikidata\\wikidata_labels_en_{lid}.jsonl')
        print("map size: ", len(mapping))
        for qid in mapping:
            time.sleep(2)
            entry = mapping[qid]
            generate([entry[0], entry[1]], qid, lid, languages[lid])
