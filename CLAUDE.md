# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目用途

学习项目：vLLM 部署与 OpenAI 兼容 API 的各种操作（chat completions、function calling、LoRA 热挂载、SFT 模型服务）。根目录的 `test_*.py` **不是 pytest 测试**，而是可直接运行的脚本，分两类：

- **服务端脚本**（阻塞运行，起一个 OpenAI 兼容 HTTP 服务在 `localhost:8000`）：`test_vllm_qwen3-0.6b.py`（基座）、`test_vllm_qwen3-0.6b-lora.py`（基座 + LoRA 热挂载）、`test_vllm_qwen3-0.6b-SFT.py`（SFT 全量权重）
- **客户端脚本**（需先起服务端）：`test_openai.py`（基础对话）、`test_fcall_basic.py`（function calling / tools）

## 运行方式

```bash
conda activate ai-gpu   # vLLM 0.10.2 所在环境

# 终端 1：起服务（一次只起一个，见下方显存约束）
python test_vllm_qwen3-0.6b.py        # 或 -lora / -SFT 变体

# 终端 2：跑客户端
python test_openai.py
python test_fcall_basic.py

# 验证服务与显存
curl http://localhost:8000/v1/models
nvidia-smi
```

模型与适配器路径在 `/home/dupengair/shared/LLM/Fine-tuning/` 下（脚本内 `main_path` 变量），三个服务端脚本各自 `--served-model-name` 为 `qwen3-0.6b`、`qwen3-0.6b-lora`、`qwen3-0.6b-sft`，客户端请求体的 `model` 字段必须与之匹配。

## 硬性约束（RTX 4050 Laptop 6GB + WSL2）

1. `os.environ["VLLM_USE_V1"] = "0"` 必须保留且在 `import vllm` 之前（envs.py 在 import 时固化默认值，V1 在此机器未验证）。
2. `--max-model-len 4096` 必须显式给：Qwen3 原生 40960 的 KV cache 预算在 6GB 上直接 OOM。
3. 同一时刻只允许一个引擎：6GB 放不下第二个引擎的权重（~1.12GB）。LoRA 版是同进程热挂载（安全），绝不能同时起两个服务进程。
4. 换服务前：Ctrl+C 后用 `nvidia-smi` 确认显存归零再起下一个（WSL2 偶发进程残留）。
5. OOM 应急三档依次加：`--gpu-memory-utilization` 0.7→0.6 → `--enforce-eager` → `--max-model-len 2048 --max-num-seqs 4`。

## 服务端脚本的关键模式

三个服务端脚本采用 `vllm serve` 官方同款组装（不要回退到手搓 Namespace + 外挂 LLM 的老写法，在 vLLM ≥0.10 已死，详见 docs 文档）：

```python
parser = make_arg_parser(FlexibleArgumentParser(...))   # 注册全部前端+引擎参数
args = parser.parse_args([...])                          # 参数意图逐项平移为 CLI 列表
validate_parsed_serve_args(args)
uvloop.run(run_server(args))                             # 内部：建引擎 → build_app → init_app_state → serve_http
```

引擎注入通道是 `app.state.engine_client`，不是 args；`args.llm_engine` 是无人读取的死代码。

## 客户端脚本要点

- 通过 `openai` SDK 指向 `http://localhost:8000/v1`，`api_key="EMPTY"` 即可。
- Qwen3 默认开思维链，对比/工具调用测试时用 `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` 关闭。
- vLLM 自动套用模型目录里的 chat template，用 `/v1/chat/completions` 而非 `/v1/completions`。
- LoRA 微调产物只有 `qwen3-0.6b_Lora` 可服务；QLoRA / AdaLora / prompt 系（Prefix/Prompt/PTuningV2）产物 vLLM 不支持。

## 工具与文档

- **代码分析用 CodeGraph**：本仓库有 `.codegraph/` 索引，分析 vLLM 内部实现（如 `run_server`、`build_app`、`from_cli_args` 的调用链）时优先用 `codegraph_explore`，而不是 grep + 逐文件读。
- `docs/vLLM部署报错分析与改造方案.md` 是核心参考：记录了 0.10.2 服务端架构（`run_server` 链路）、手搓 Namespace 为何必炸（三层坑）、6GB 约束的由来，以及踩坑归档（坑 20、21）。
- 回答问题使用中文
- 需要记录的文档放到 ./docs 下面
