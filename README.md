# contextual-retrieval-re
对 Anthropic 提出的 [contextual-retrieval](https://www.anthropic.com/news/contextual-retrieval) 的简单复现。官方代码在[这里](https://github.com/anthropics/anthropic-cookbook/tree/main/skills/contextual-embeddings)。

## Dependence

```
pip install openai tqdm langchain langchain_community langchain_huggingface faiss-cpu
```

## Usage
首先为 chunk 添加上下文，这里使用的是 `deepseek-chat` 的 API。请在 `context_generation.py` 中修改 `API_KEY`，如果想使用其他模型，则修改对应的 `API_KEY` 和 `CHAT_MODEL` 即可。
```bash
python context_generation.py
```
我生成的结果放在了 `/data/context_chunks_deepseek.json`。
注意这一步如果 CPU 内存不够可能会报错，如果发生这种情况考虑把所有的数据分批次使用，然后merge到一起。

得到包含上下文信息的chunk后，运行 `main.py`，这里我们使用的是 `faiss` 向量数据库，embedding 模型使用的是 [bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)。
运行 baseline （没有使用 contextual-retrieval)
```
python main.py --database_path data/codebase_chunks.json --db_path faiss_index --top_k 5
python main.py --database_path data/codebase_chunks.json --db_path faiss_index --top_k 10 --load_local_db
python main.py --database_path data/codebase_chunks.json --db_path faiss_index --top_k 20 --load_local_db
```
然后是使用了 contextual-retrieval 的情况
```
python main.py --database_path data/context_chunks_deepseek.json --db_path faiss_index_context --top_k 5
python main.py --database_path data/context_chunks_deepseek.json --db_path faiss_index_context --top_k 10 --load_local_db
python main.py --database_path data/context_chunks_deepseek.json --db_path faiss_index_context --top_k 20 --load_local_db
```
## Result
| Recall@K   | w/o context  | w/ context   |
|-------|-------|-------|
 5|     74.6   | 83.1
 10|    82.7   | 88.7
 20|    86.7   | 91.9




