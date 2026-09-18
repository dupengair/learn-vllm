# vLLM 部署学习仓库（Qwen3-0.6B × OpenAI 兼容 API）

在 **RTX 4050 Laptop 6GB + WSL2** 上使用 **vLLM 0.10.2** 部署 Qwen3-0.6B 的学习仓库：覆盖 OpenAI 兼容 API 的基础对话、function calling / tools、LoRA 热挂载、SFT 全量权重服务四种玩法，以及在小显存机器上踩出来的全部坑与源码级分析。

## 环境与硬件

| 项 | 值 |
|---|---|
| GPU | RTX 4050 Laptop，**6GB**（权重 ~1.12GB，KV cache 预算极紧） |
| 系统 | Windows + WSL2 |
| Python 环境 | conda 环境 `ai-gpu`（vLLM 0.10.2） |
| 模型 | Qwen3-0.6B 基座 + 自训 LoRA / SFT 变体 |

## 目录结构

```
vllm/
├── README.md                            # 本文件
├── CLAUDE.md                            # Claude Code 协作指引（仓库约定与硬性约束）
│
│  ── 服务端脚本（阻塞运行，起 OpenAI 兼容服务于 localhost:8000）──
├── test_vllm_qwen3-0.6b.py              # 基座模型服务（served-model-name: qwen3-0.6b）
├── test_vllm_qwen3-0.6b-lora.py         # 基座 + LoRA 热挂载（served-model-name: qwen3-0.6b-lora）
├── test_vllm_qwen3-0.6b-SFT.py          # SFT 全量权重服务（served-model-name: qwen3-0.6b-sft）
│
│  ── 客户端脚本（需先起对应服务端）──
├── test_openai.py                       # 基础对话（openai SDK 指向 localhost:8000/v1）
├── test_fcall_basic.py                  # function calling / tools（含函数映射表白名单执行）
│
└── docs/
    ├── vLLM部署报错分析与改造方案.md     # 服务端架构（run_server 链路）、手搓 Namespace 为何必炸、坑 1~21
    └── FunctionCall调试记录.md           # function calling 六个问题的完整调试（问题 1~6、坑 22~26）
```

**注意**：根目录的 `test_*.py` 不是 pytest 测试，而是可直接运行的脚本，分"服务端"（阻塞式 HTTP 服务）与"客户端"（请求该服务）两类。

## 三个服务端脚本的差异

均采用 `vllm serve` 官方同款组装（`make_arg_parser` → `validate_parsed_serve_args` → `uvloop.run(run_server)`），不要回退到手搓 Namespace + 外挂 LLM 的老写法（vLLM ≥0.10 已死，详见 docs）。

| 脚本 | `--model` | `--served-model-name` | 特有参数 |
|---|---|---|---|
| `test_vllm_qwen3-0.6b.py` | `model/Qwen3-0.6B` | `qwen3-0.6b` | — |
| `test_vllm_qwen3-0.6b-lora.py` | `model/Qwen3-0.6B` | `qwen3-0.6b-lora` | `--enable-lora --max-loras 1 --max-lora-rank 8 --lora-modules` |
| `test_vllm_qwen3-0.6b-SFT.py` | `training/qwen3-0.6b_SFT/lora_adapter` | `qwen3-0.6b-sft` | — |

- 客户端请求体的 `model` 字段必须与 `--served-model-name` 匹配，起错变体会 404。
- LoRA 微调产物只有 `qwen3-0.6b_Lora` 可被 vLLM 服务；QLoRA / AdaLora / prompt 系（Prefix/Prompt/PTuningV2）产物 vLLM 不支持。
- 需要 function calling 的服务端必须带 `--enable-auto-tool-choice --tool-call-parser hermes`（Qwen2.5/Qwen3 系列用 `hermes`）。

## 快速开始

```bash
conda activate ai-gpu

# 终端 1：起服务（一次只起一个，见下方硬性约束）
python test_vllm_qwen3-0.6b.py        # 或 -lora / -SFT 变体

# 终端 2：跑客户端
python test_openai.py
python test_fcall_basic.py

# 验证服务与显存
curl http://localhost:8000/v1/models
nvidia-smi
```

模型与适配器路径写死在脚本内 `main_path` 变量（本机为 `/home/dupengair/shared/LLM/Fine-tuning/`），克隆到其他机器需自行修改。

## 硬性约束（6GB 生存法则）

1. `os.environ["VLLM_USE_V1"] = "0"` 必须保留且在 `import vllm` 之前（envs.py 在 import 时固化默认值，V1 在此机器未验证）。
2. `--max-model-len 4096` 必须显式给：Qwen3 原生 40960 的 KV cache 预算在 6GB 上直接 OOM。
3. 同一时刻只允许一个引擎：6GB 放不下第二个引擎的权重。LoRA 版是同进程热挂载（安全），绝不能同时起两个服务进程。
4. 换服务前：Ctrl+C 后用 `nvidia-smi` 确认显存归零再起下一个（WSL2 偶发进程残留）。
5. OOM 应急三档依次加：`--gpu-memory-utilization` 0.7→0.6 → `--enforce-eager` → `--max-model-len 2048 --max-num-seqs 4`。

## Function Calling 要点（V0 引擎，vLLM 0.10.2）

- 服务端：`--enable-auto-tool-choice` + `--tool-call-parser hermes`（纯前端解析参数，不占显存）。
- 客户端：传 `tools` 不传 `tool_choice` 时协议层自动补 `"auto"`；模型侧由 hermes parser 解析 `<tool_call>{json}</tool_call>` 输出。
- `tool_choice="required"` / guided decoding 等结构化输出在 0.10.2 的 **V0 引擎上已静默移除**（参数被接受但无人执行），本机禁用。
- 小模型（0.6B 级）选工具全靠 description 文本：相似工具必须写互斥的正反条件；工具调用场景建议降温至 0~0.2。
- 执行侧永远用 `{函数名: 函数对象}` 映射表查表调用，禁止 `eval` 模型输出的函数名。

## 文档索引

| 文档 | 内容 |
|---|---|
| [docs/vLLM部署报错分析与改造方案.md](docs/vLLM部署报错分析与改造方案.md) | 0.10.2 服务端架构（`run_server` 链路）、手搓 Namespace 三层坑、6GB 约束由来、坑 1~21 归档 |
| [docs/FunctionCall调试记录.md](docs/FunctionCall调试记录.md) | function calling 从 400 到成功的六个问题全记录（含 vLLM 源码定位、离线渲染 prompt、V0/V1 能力差异），坑 22~26 归档 |
