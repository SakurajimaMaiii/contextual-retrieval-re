import json
import argparse

from openai import OpenAI

API_KEY = ""
API_BASE = "https://api.deepseek.com"
CHAT_MODEL = "deepseek-chat"
CONTEXT_PROMPT = f"""<document> 
{{WHOLE_DOCUMENT}} 
</document> 
Here is the chunk we want to situate within the whole document 
<chunk> 
{{CHUNK_CONTENT}} 
</chunk> 
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else. """

client = OpenAI(api_key=API_KEY, base_url=API_BASE)


parser = argparse.ArgumentParser()
parser.add_argument("--database_path", type=str, default="data/codebase_chunks.json")
parser.add_argument("--output_file", type=str,default="data/context_chunks_deepseek.json")
args = parser.parse_args()


with open(args.database_path,"r") as f:
    transformed_dataset = json.load(f)

for doc_id,doc in enumerate(transformed_dataset):
    full_content = doc["content"]
    chunks = doc["chunks"]
    for chunk_id,c in enumerate(chunks):
        _data = {}
        chunk_content = c["content"]
        prompt = CONTEXT_PROMPT.format(WHOLE_DOCUMENT=full_content,CHUNK_CONTENT=chunk_content)
        response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "user", "content":prompt},
                ],
                stream=False
            )
        
        response = response.choices[0].message.content
        transformed_dataset[doc_id]["chunks"][chunk_id]["content"] = response + chunk_content
        print(f"Doc {doc_id}, chunk {chunk_id}")

with open(args.output_file,"w") as f:
    json.dump(transformed_dataset, f, indent=4)
print(f"Save to {args.output_file}")
        