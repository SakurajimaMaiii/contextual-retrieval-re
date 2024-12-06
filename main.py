import json
import argparse

from tqdm import tqdm
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


parser = argparse.ArgumentParser()
parser.add_argument("--database_path", type=str, default="data/codebase_chunks.json")
parser.add_argument("--evaluation_data_path", type=str,default="data/evaluation_set.jsonl")
parser.add_argument("--embedding_model",type=str,default="./bge-large-en-v1.5")
parser.add_argument("--load_local_db",action="store_true")
parser.add_argument("--db_path",type=str,default="./faiss_index_context")
parser.add_argument("--top_k",type=int,default=5)
args = parser.parse_args()

# loda dataset
with open(args.database_path,"r") as f:
    transformed_dataset = json.load(f)

with open(args.evaluation_data_path,"r") as file:
    eval_data = [json.loads(line) for line in file]

database = []

for doc in transformed_dataset:
    chunks = doc["chunks"]
    doc_id = doc["original_uuid"]
    for c in chunks:
        _data = {}
        _data["content"] = c["content"]
        _data["ids"] = [doc_id,c["original_index"]]
        database.append(_data)

print(f"database size:{len(database)}")

documents = [Document(page_content=c["content"],metadata={"ids":c["ids"]}) for c in database]

embeddings = HuggingFaceEmbeddings(model_name=args.embedding_model)

if args.load_local_db:
    print(f"Load local db from {args.db_path}")
    vector_store = FAISS.load_local(
    args.db_path, embeddings, allow_dangerous_deserialization=True
)
else:
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(args.db_path)
    print(f"Save db to {args.db_path}")



print("===> Start evaluation")
total = 0
correct = 0
for query_item in tqdm(eval_data):
    query = query_item["query"]
    golden_chunk_uuids = query_item['golden_chunk_uuids'][0]
    r = vector_store.similarity_search(query,k=args.top_k)
    retrieval_ids = [_r.metadata["ids"] for _r in r]
    if golden_chunk_uuids in retrieval_ids:
        correct += 1
    total += 1

recall = correct / total * 100
print(f"Pass@{args.top_k} is {recall:.2f} %")


