# contextual-retrieval-re

A simple reproduction of [contextual-retrieval](https://www.anthropic.com/news/contextual-retrieval) proposed by Anthropic. The official code is available [here](https://github.com/anthropics/claude-cookbooks/tree/main/capabilities/contextual-embeddings). The LLM used is `deepseek-chat`, and the embedding model used is `bge-large-en-v1.5`. It can run without GPU.

## Dependencies

```
pip install openai tqdm langchain langchain_community langchain_huggingface faiss-cpu
```

## Usage
First, add context to the chunks. Here, we use the `deepseek-chat` API.
```bash
export OPENAI_API_KEY=XXX
export OPENAI_BASE_URL=https://api.deepseek.com
python context_generation.py --chat_model deepseek-chat
```
To use other models, such as `gpt-4o`, simply modify the corresponding `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `chat_model` accordingly.

The results I generated are stored in `/data/context_chunks_deepseek.json`. [DeepSeek](https://www.deepseek.com/) uses disk caching technology; if the request prefix is the same, the cost is one-tenth of the normal input. In this task scenario, this can save a significant number of tokens. Please refer to the [documentation](https://api-docs.deepseek.com/guides/kv_cache) for details.

After obtaining the chunks with contextual information, run `main.py`. Here we use the `faiss(cpu)` vector database and the [bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) embedding model.

Run the baseline (without contextual retrieval)
```
python main.py --database_path data/codebase_chunks.json --db_path faiss_index --top_k 5
python main.py --database_path data/codebase_chunks.json --db_path faiss_index --top_k 10 --load_local_db
python main.py --database_path data/codebase_chunks.json --db_path faiss_index --top_k 20 --load_local_db
```
Then run the version with contextual retrieval
```
python main.py --database_path data/context_chunks_deepseek.json --db_path faiss_index_context --top_k 5
python main.py --database_path data/context_chunks_deepseek.json --db_path faiss_index_context --top_k 10 --load_local_db
python main.py --database_path data/context_chunks_deepseek.json --db_path faiss_index_context --top_k 20 --load_local_db
```
Note that using the `faiss` database may cause errors if CPU memory is insufficient. If this happens, consider processing all data in batches and then merging them.

## Result
| Recall@K   | w/o context  | w/ context   |
|-------|-------|-------|
| 5     | 74.6  | 83.1  |
| 10    | 82.7  | 88.7  |
| 20    | 86.7  | 91.9  |

It can be seen that after adding contextual information to the chunks, recall improves significantly.
