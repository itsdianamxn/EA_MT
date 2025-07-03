import json
import time
from google import genai
client = genai.Client()

# PROJECT_ROOT = "/content/drive/MyDrive/"
PROJECT_ROOT = "."

def load_entity_mapping(file_path):
    mapping = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for l in f:
            entry = json.loads(l)
            id = entry.get("id", "").strip()
            source = entry.get("source", "").strip()
            target = entry.get("target", "").strip()
            if target == "":
                continue
            mapping[id] = [source, target]
    return mapping

def generate(lang_id, lang_name, mapping):
    line_no = 0
    with open(f"{PROJECT_ROOT}/Train/fr/train.jsonl", 'r', encoding='utf-8') as in_file, \
            open(f"{PROJECT_ROOT}/Train/{lang_id}/train.jsonl", 'a', encoding='utf-8') as out_file:

        for line in in_file:
            line_no += 1
            if line_no < 1:  # skip these lines
                continue
            try:
                entry = json.loads(line)
                source = entry.get("source", "")
                ids = entry.get("entities", "")
                generated = entry.get("from", "")
                if not source or generated == "generated":
                    continue
                entity_str = []
                for entity in ids:
                    if entity not in mapping:
                        continue
                    entity_str.append(f'the named entity "{mapping[entity][0]}" is translated as: "{mapping[entity][1]}"')
                if not entity_str:
                    continue
                prompt = (f"You are an expert translator. Translate the following sentence from English to {lang_name}," +
                          f' where {" and ".join(entity_str)}. Generate only one translation, ' +
                          f"with no additional information or explanations:\n{source}")
                print("Prompt:", prompt)
                retry_cnt = 0
                while retry_cnt < 3:
                    try:
                        retry_cnt += 1
                        response = client.models.generate_content(
                            model="gemini-2.0-flash-001",
                            contents=prompt,
                        )
                        break
                    except Exception as e:
                        print(f"Retrying {retry_cnt} for line #{line_no}")
                        time.sleep(10)


                print(f'{line_no}: {response.text}')
                entry["target_locale"] = lang_id
                entry["target"] = response.text.strip()
                json.dump(entry, out_file, ensure_ascii=False)
                out_file.write('\n')
                time.sleep(2)  # wait a second...
            except Exception as e:
                print(f"Error: {e} for line #{line_no}: {line}")


if __name__ == '__main__':
    languages = {
        "ro": "Romanian",
        #        "fr": "French",
        #         "it": "Italian",
        #         "de": "German",
        #         "es": "Spanish",
        #         "ja": "Japanese",
    }
    mapping = load_entity_mapping(
        f'C:\\Users\\diana\\Desktop\\Anul III\\Licenta\\PROIECT\\Scripts\\Wikidata\\wikidata_labels_en_ro.jsonl')
    print("map size: ", len(mapping))
    for lid in languages:
        generate(lid, languages[lid], mapping)
