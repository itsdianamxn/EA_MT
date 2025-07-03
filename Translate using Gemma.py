import json
import time
from google import genai
client = genai.Client()

#PROJECT_ROOT = "/content/drive/MyDrive/"
PROJECT_ROOT = "."

def translate(lang_id, lang_name):
    line_no = 0
    with open(f"{PROJECT_ROOT}/Test/{lang_id}_JP.jsonl", 'r', encoding='utf-8') as in_file, \
         open(f"{PROJECT_ROOT}/Results/{lang_id}_trans_gemma.jsonl", 'a', encoding='utf-8') as out_file:
        for line in in_file:
            line_no += 1
            if line_no < 4084: # skip these lines
                continue
            try:
                entry = json.loads(line)
                source = entry.get("source", "")
                if not source:
                    continue
                prompt = (f"You are an expert translator. Translate the following sentence from English to {lang_name}. " +
                          f"Ensure that all named entities are accurately translated." +
                          f"Provide only one translation, no additional explanation, no phonetic transcription and nothing else:\n{source}")
                response = client.models.generate_content(
                    model="gemma-3-1b-it",
                    contents=prompt,
                )
                print(f'{line_no}: {response.text}')
                entry["targets"].append(response.text.strip())
                json.dump(entry, out_file, ensure_ascii=False)
                out_file.write('\n')
                time.sleep(2)  # wait a second...
            except Exception as e:
                print(f"Error: {e} for line #{line_no}: {line}")
            
if __name__ == '__main__':
    languages = {
#         "ar": "Arabic",
#        "fr": "French",
#         "it": "Italian",
#         "de": "German",
#         "es": "Spanish",
         "ja": "Japanese",
    }
    for lid in languages:
        translate(lid, languages[lid])
