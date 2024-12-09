# contextual-retrieval-re
对 Anthropic 提出的 [contextual-retrieval](https://www.anthropic.com/news/contextual-retrieval) 的简单复现。官方代码在[这里](https://github.com/anthropics/anthropic-cookbook/tree/main/skills/contextual-embeddings)。使用的 LLM 为 `deepseek-chat`，使用的 embedding 模型为 bge-large-en-v1.5，无需使用 GPU 即可运行。

## Dependence

```
pip install openai tqdm langchain langchain_community langchain_huggingface faiss-cpu
```

## Usage
首先为 chunk 添加上下文，这里使用的是 `deepseek-chat` 的 API。
```bash
export OPENAI_API_KEY=XXX
export OPENAI_BASE_URL=https://api.deepseek.com
python context_generation.py --chat_model deepseek-chat
```
想使用其他模型，如 `gpt-4o` 等，修改对应的 `OPENAI_API_KEY` ， `OPENAI_BASE_URL`，`chat-model` 即可。
我生成的结果放在了 `/data/context_chunks_deepseek.json`。[DeepSeek](https://www.deepseek.com/) 使用了硬盘缓存技术，如果请求前缀相同，价格是正常输入的十分之一。在当前任务场景下，可以节省非常多的 token 消耗，请参考[文档](https://api-docs.deepseek.com/guides/kv_cache)。

得到包含上下文信息的chunk后，运行 `main.py`，这里我们使用的是 `faiss(cpu)` 向量数据库，embedding 模型使用的是 [bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)。
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
注意使用 `faiss` 数据库如果 CPU 内存不够可能会报错，如果发生这种情况考虑把所有的数据分批次使用，然后merge到一起。
## Result
| Recall@K   | w/o context  | w/ context   |
|-------|-------|-------|
 5|     74.6   | 83.1
 10|    82.7   | 88.7
 20|    86.7   | 91.9

可以看到在给 chunk 添加上下文信息后，recall 明显提升。


