# transformers 5.17.0 导致 vLLM 0.10.2 启动失败分析 —— 跨仓库的依赖连锁反应

> 对应脚本：`test_vllm_qwen3-0.6b.py`（以及 -lora / -SFT 两个变体，同锅）
> 现象：`AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended`，同一错误连打两遍（APIServer 主进程一遍、SpawnProcess-1 引擎子进程一遍），服务起不来
> 关联：怀疑由 langchain 仓库的 vecemb 修复引发 —— **怀疑完全属实**，记录见 `langchain/docs/06-vecemb-bert-unexpected.md`
> 结论先行：**9 月 22 日修 vecemb 时安装的 sentence-transformers 6.1.0 强制要求 transformers≥5.0.0，pip 把 transformers 从 4.x 连带升到 5.17.0；而 transformers 5.0 删除了 vLLM 0.10.2 正在调用的 `all_special_tokens_extended` 属性。修复 = 降级到三方兼容点 `transformers==4.57.6 + sentence-transformers==5.1.0`（已 dry-run 验证无连锁冲突）。本文不改任何源码，修复命令见第 6 节。**
> 所有数据均来自本机实测（`ai-gpu` 环境 site-packages 与 PyPI wheel 下载比对），行号基于本机 vLLM 0.10.2 / transformers 5.17.0。

## 1. 结论速览

| 问题 | 判定 | 依据 |
|------|------|------|
| 和 vecemb 修复有关吗？ | ✅ **直接因果** | 9/22 20:05 `sentence-transformers-6.1.0`、`transformers-5.17.0`、`tokenizers-0.23.2` 三个 dist-info 同一分钟落盘；20:22 langchain 06 文档写成 |
| transformers 是被谁升的？ | sentence-transformers 6.1.0 | 其元数据强制 `transformers>=5.0.0,<6.0.0`；bge 嵌入修复需要装它 |
| vLLM 为什么崩？ | 调了被删除的 API | `vllm/transformers_utils/tokenizer.py:99` 读 `tokenizer.all_special_tokens_extended`；该属性在 transformers 5.17.0 源码中全文检索 **0 处**（4.x 有，5.0 删） |
| pip 为什么没拦住？ | vLLM 依赖**欠约束** | vllm 0.10.2 只声明 `transformers>=4.55.2` **无上界**；5.17.0 同时满足两个包的元数据约束，pip 认为没冲突——元数据兼容 ≠ API 兼容 |
| vecemb 侧需要回滚吗？ | ❌ 不需要 | 降级方案对 bge-small-zh-v1.5 无影响，sentence-transformers 5.x 完全支持它 |

## 2. 因果链：一次修复如何引爆另一个仓库

### 2.1 时间线（site-packages dist-info 目录时间 + 文档 mtime）

| 时间 | 事件 |
|------|------|
| 08-27 01:06 | `vllm-0.10.2` 安装（彼时 transformers 为 4.5x，一切正常） |
| **09-22 20:05** | **`sentence-transformers-6.1.0` + `transformers-5.17.0` + `tokenizers-0.23.2` 同一分钟落盘** ← vecemb 修复装包 |
| 09-22 20:22 | `langchain/docs/06-vecemb-bert-unexpected.md` 写成（换 bge-small-zh-v1.5 方案） |
| 09-23 14:04 | 跑 `test_vllm_qwen3-0.6b.py` → AttributeError，服务起不来 |

三个包同一时间戳落盘，是 pip 一次解析安装的典型指纹：装 sentence-transformers 6.1.0 时，pip 发现它要求 `transformers>=5.0.0,<6.0.0`，而 vllm 0.10.2 的约束是 `transformers>=4.55.2`（无上界）——**5.17.0 同时满足两者**，于是 pip "合法地"执行了升级，连带 tokenizers 也升到 0.23.2。

### 2.2 依赖元数据实测（本机 `importlib.metadata`）

| 包 | 版本 | 对 transformers 的要求 |
|---|---|---|
| vllm | 0.10.2 | `>=4.55.2`（**无上界** ← 问题根源） |
| sentence-transformers | 6.1.0 | `>=5.0.0,<6.0.0`（**强制 5.x** ← 升级推手） |
| langchain-huggingface | 1.2.2 | 核心依赖**不含** transformers 约束（`>=5.0.0` 只在 `extra=='full'` 可选组里，未生效） |

**判读：pip 没有出错，它严格遵守了元数据。错在 vllm 0.10.2 的依赖声明没写上界**——代码里调用的 API 在 5.0 被删，元数据却宣称兼容一切 ≥4.55.2 的版本。这类"元数据兼容、API 不兼容"是跨包升级事故的最常见形态。

### 2.3 源码对撞点

vLLM 侧（0.10.2，`vllm/transformers_utils/tokenizer.py:88-101`，`get_cached_tokenizer`——把慢 tokenizer 的重复计算属性缓存起来）：

```python
def get_cached_tokenizer(tokenizer: AnyTokenizer) -> AnyTokenizer:
    cached_tokenizer = copy.copy(tokenizer)

    tokenizer_all_special_ids = tokenizer.all_special_ids
    tokenizer_all_special_tokens = tokenizer.all_special_tokens
    tokenizer_all_special_tokens_extended = (
        tokenizer.all_special_tokens_extended)        # ← :99，transformers 5.x 已删除此属性
    ...
```

transformers 侧：

- **4.57.6**（4.x 末版）：`tokenization_utils_base.py:1164` 定义为 `@property all_special_tokens_extended`，返回 `list[Union[str, AddedToken]]`，正常无警告（从 PyPI wheel 下载比对确认，另在 :917、:1191 还有两处内部使用）；
- **5.17.0**：整个包 grep 该名字 **0 处**——5.0 大清理 tokenizer API 时删除。

于是 `tokenizer.all_special_tokens_extended` 落入 `tokenization_utils_base.py:1308` 的 `__getattr__` 兜底，抛出 `AttributeError: Qwen2Tokenizer has no attribute ...`，末尾那句 `Did you mean: 'num_special_tokens_to_add'?` 是这个 `__getattr__` 用 difflib 给的最近似建议——纯属误导，跟真正的修复方向无关。

### 2.4 为什么同一个错打两遍

vLLM 的多进程架构里 tokenizer 初始化了**两次**，两条路都经过 `get_cached_tokenizer`：

1. **APIServer 主进程**：`run_server` → `build_async_engine_client` → `MQClient.__init__`（`engine/multiprocessing/client.py:103`）→ `init_tokenizer_from_configs` → 炸。client 侧要 tokenizer 做请求预处理（chat template 渲染、计数）；
2. **引擎子进程**（SpawnProcess-1）：`MQLLMEngine.from_vllm_config` → `LLMEngine.__init__`（`llm_engine.py:237`）→ `_init_tokenizer` → 同函数再炸。engine 侧要 tokenizer 做 detokenize。

两遍 traceback 内容一致不是"两个 bug"，是同一颗雷在两个进程各踩一次。日志里主进程炸完子进程还继续跑到同一行，是因为 MQLLMEngine 是 spawn 出来的独立进程，不共享主进程的失败状态。

## 3. `all_special_tokens_extended` 是什么、为什么值得 vLLM 缓存它

- `all_special_tokens` 返回**纯字符串**列表（如 `['<|im_end|>', '<|endoftext|>']`）；
- `all_special_tokens_extended` 返回 **`AddedToken` 对象**列表，额外携带 `lstrip/rstrip/single_word/normalized` 等行为元数据——vLLM 缓存它是因为这个 property 每次访问都要遍历 `special_tokens_map_extended` 重新组装，慢 tokenizer 上是热路径（函数 docstring 自己写着 "leading to a significant slowdown"）；
- transformers 5.0 的 tokenizer 大清理把 extended 系属性移除，统一收敛到非 extended 版本。

vLLM 上游后续版本改掉了这个调用，但 0.10.2 没赶上——所以**这个坑的边界是：本环境里 vllm 0.10.2 与 transformers≥5.0 互斥**。

## 4. 修复方案比选

| 方案 | 做法 | 判定 |
|------|------|------|
| **A. 降级到三方兼容点** | `transformers==4.57.6 + sentence-transformers==5.1.0` | ✅ **推荐**，一次命令，两个仓库都活 |
| B. 环境隔离 | langchain 嵌入（CPU 即可）单独建轻量 conda env，ai-gpu 专供 vLLM | ✅ 长期最干净，但要维护两套环境，先不必 |
| C. 升级 vLLM 到适配 transformers 5.x 的版本 | 大版本升级 | ❌ 0.10.2 + V0 + 6GB + WSL2 是精心调过的组合（VLLM_USE_V1=0、max-model-len 4096…），升级连带 torch/CUDA 连锁重装，学习仓库不稳妥 |
| D. monkey-patch 补属性 | 脚本里给 Qwen2Tokenizer 动态补 property | ❌ 治标不治本：三个服务端脚本都要加、掩盖真实版本关系、且 5.x 后面可能还有第二个对撞点。仅作理解原理用 |

## 5. 方案 A 的兼容性论证（全部已验证）

选 `transformers==4.57.6` 与 `sentence-transformers==5.1.0` 这个组合，三方约束同时满足：

| 消费方 | 对 transformers 的要求 | 4.57.6 满足？ | 验证方式 |
|---|---|---|---|
| vllm 0.10.2 | `>=4.55.2` | ✅ | 本机元数据 |
| sentence-transformers 5.1.0 | `>=4.41.0,<5.0.0` | ✅ | PyPI wheel METADATA 实测 |
| vLLM 代码调用 | `all_special_tokens_extended` 存在 | ✅ | 4.57.6 wheel 源码 :1164 确认存在且无弃用警告 |

额外两个好处：

1. sentence-transformers 5.1.0 的 `transformers<5.0.0` **上界反而成了保护性钉子**——以后在 ai-gpu 里误装任何要求 transformers≥5 的包，pip 会立刻报解析冲突报警，而不是静默升级后再炸；
2. 对 langchain 侧零影响：bge-small-zh-v1.5 是标准 BERT 架构 + sentence-transformers 配置，5.x 完全支持；langchain-huggingface 1.2.2 核心依赖不含这两者的版本约束。

## 6. 修复命令（手动执行）

```bash
conda activate ai-gpu
pip install "transformers==4.57.6" "sentence-transformers==5.1.0"
```

**已在本机 `--dry-run` 验证**，实际变动仅 4 个包、全部在兼容范围内、不触碰 vllm/torch：

```
Would install huggingface_hub-0.36.2 sentence-transformers-5.1.0 tokenizers-0.22.2 transformers-4.57.6
```

- `tokenizers 0.23.2 → 0.22.2`：transformers 4.57.6 要求 tokenizers<0.23，正常连带降级；
- `huggingface_hub → 0.36.2`：满足 langchain-huggingface 的 `>=0.33.4,<2.0.0`。

## 7. 验证清单

执行降级后依次检查：

1. 版本钉住：
   ```bash
   python -c "import transformers, sentence_transformers; print(transformers.__version__, sentence_transformers.__version__)"
   # 期望: 4.57.6 5.1.0
   ```
2. 被删属性复活（模拟 vLLM 的调用点）：
   ```bash
   python -c "
   from transformers import AutoTokenizer
   t = AutoTokenizer.from_pretrained('/home/dupengair/shared/LLM/Fine-tuning/model/Qwen3-0.6B')
   print(t.all_special_tokens_extended)"
   # 期望: 打印 AddedToken 列表（如 [<|im_end|>, <|endoftext|>]），不再 AttributeError
   ```
3. vLLM 服务恢复：`python test_vllm_qwen3-0.6b.py` 起服务 → 另一终端 `curl http://localhost:8000/v1/models`；
4. langchain 侧回归：重跑 `test_langchain_vecemb.py`，确认 bge 嵌入与检索不受降级影响（第 1 名仍应命中 LangChain 片段）；
5. 元数据体检：`pip check` 无冲突输出。

## 8. 防复发（坑 22 归档）

**坑 22：vllm 0.10.2 的 transformers 依赖无上界（`>=4.55.2`），任何把 transformers 推到 ≥5.0 的安装都会静默炸掉服务启动**——本例推手是 sentence-transformers 6.1.0（要求 transformers≥5.0.0），随 langchain vecemb 修复于 9/22 入侵；爆点在 `get_cached_tokenizer` 读 `all_special_tokens_extended`（transformers 5.0 已删，4.x 末版 4.57.6 仍在），且因 client/engine 两进程各初始化一次 tokenizer，同一错误连打两遍。防护三件套：

1. ai-gpu 里装任何新包，先 `pip install --dry-run <pkg>` 看连带变动——出现 transformers/torch/vllm 字样时停下来想；
2. 维持 `sentence-transformers==5.1.0`（其 `transformers<5` 上界是保护性钉子），不要升 6.x；
3. 根治靠环境隔离：哪天 langchain 生态必须用 transformers 5.x 了，给嵌入单独建 CPU 轻量 env，ai-gpu 专供 vLLM。

## 9. 一句话总结

**vLLM 没坏、Qwen3 没坏、脚本也没坏——是 9/22 修 vecemb 时 sentence-transformers 6.1.0 借 pip 之手把 transformers 升到了 5.17.0，删掉了 vLLM 0.10.2 正在调用的 tokenizer 属性；pip 之所以放行，是因为 vllm 的依赖声明只写了下界没写上界。降回 `transformers==4.57.6 + sentence-transformers==5.1.0` 三方兼容点，两个仓库继续共存。**
