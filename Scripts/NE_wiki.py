import json
import requests
import os
def extract_ids(path, ):
    ids = set(())
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if 'wikidata_id' in data:
                ids.add(data['wikidata_id'])
            else:
                if 'entities' in data:
                    for entity in data['entities']:
                        ids.add(entity)
    return list(ids)

def extract_data(ids, lang_source, lang_target):
    out_path = f"wikidata_labels_{lang_source}_{lang_target}.jsonl"
    seen = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf‑8") as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["id"])
                except Exception:
                    pass
    with open(out_path, "a", encoding="utf-8") as f:
        for i in range(0, len(ids), 50):
            batch = ids[i:i + 50]
            print(f"Processing batch {i // 50 + 1} with {batch}")
            params = {
                'action': 'wbgetentities',
                'format': 'json',
                'ids': '|'.join(batch),
                'languages': f"{lang_target}|{lang_source}",
                'props': 'labels'
            }
            # url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&format=json&ids={ids_string}&languages={langs}&props=labels"
            response = requests.get("https://www.wikidata.org/w/api.php", params=params)
            response.raise_for_status()
            data = response.json()
            entities = data.get('entities', {})

            for entity_id, entity_data in entities.items():
                labels = entity_data.get('labels', {})
                label_source = labels.get(lang_source, {}).get('value', '')
                label_target = labels.get(lang_target, {}).get('value', '')
                f.write(json.dumps({
                    'id': entity_id,
                    'source': label_source,
                    'target': label_target
                }, ensure_ascii=False) + '\n')
                print(f"Entity ID: {entity_id}, Source: {label_source}, Target: {label_target}")

if __name__ == '__main__':

    #path = "C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/data/references/validation/fr_FR.jsonl"
    #path = "C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/data/references/test/fr_FR.jsonl"
    path = "C:/Users/diana/Desktop/Anul III/Licenta/PROIECT/data/semeval/train/it/train.jsonl"


    ids = extract_ids(path)
    #extract_data(ids, "en", "fr")
    extract_data(ids, "en", "it")