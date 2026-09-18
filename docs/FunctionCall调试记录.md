# Function Calling 调试记录（test_fcall_basic.py）

> 2026-09-17。目标：通过 vLLM OpenAI 兼容服务调试 `tools` 参数与工具调用解析。
> 首跑（`qwen3-0.6b-lora` 服务）连出三个问题：一个 HTTP 侧拼写、一个 SDK 属性名、一个执行侧反模式（`eval`）。
> 二跑（基座 `qwen3-0.6b` 服务）新增问题 4：服务端未开启工具解析，请求在 HTTP 层被 400 拒绝（见第四节）。
> 三跑（问题 4 修复后）新增问题 5：模型不发起工具调用，`tool_calls=[]`，纯文本反问用户要日期（见第五节）。
> 四跑（应用修复 1+2+4 后）新增问题 6：`tool_choice="required"` 在 V0 引擎上约束不生效，模型输出的 `<tool_call>` 标签被 required 解析器按纯 JSON 解析而 400（见第六节）。**该错误同时证明了修复 1+2 已生效——模型已正确选定函数与参数。**
> 版本事实基于本机 `ai-gpu` 环境（vLLM 0.10.2 + openai SDK），行号为修复后当前行号。

---

## 一、问题清单与一句话结论

| # | 位置 | 问题 | 后果 | 修法 |
|---|---|---|---|---|
| 1 | `test_openai.py:9` | `"role":"sysstem"` 拼写错误 | vLLM 不识别该 role，系统提示词**静默失效**（不报错） | 改为 `"system"` |
| 2 | `test_fcall_basic.py:80` | `message.tool_call` 属性名少了 s | `AttributeError`（openai SDK 的字段是 `tool_calls`） | 改为 `message.tool_calls` |
| 3 | `test_fcall_basic.py:84`（修复前） | `eval(func_name)(**args)` 动态执行模型输出 | 幻觉函数名直接崩；本质是把模型输出当代码执行 | 字典映射 `{func_name: func}` + 查表调用 |
| 4 | 服务端 `test_vllm_qwen3-0.6b.py:21-29` | 未传 `--enable-auto-tool-choice` / `--tool-call-parser` | `400: "auto" tool choice requires ...`，请求进不到模型 | 参数列表补这两项，parser 用 `hermes`（见第四节） |
| 5 | 客户端 `test_fcall_basic.py:27-59` 等 | 两个工具描述区分度不足 + 采样偏热，0.6B 混淆工具前置条件 | `tool_calls=[]`，纯文本反问"请提供日期" | 描述写明互斥适用条件 + system 立规 + 降温，~~兜底 `tool_choice="required"`~~（后者在 V0 不可用，见第六节） |
| 6 | 客户端 `test_fcall_basic.py:76` | `tool_choice="required"` 依赖的 guided decoding 在 V0 引擎（0.10.2）不生效 | 400：required 解析器把含 `<tool_call>` 标签的输出按纯 JSON 解析失败 | 删掉 `tool_choice="required"` 回到 auto，交给已配置的 hermes parser（见第六节） |

**一句话结论**：1、2 是"字段名对不上"的表面问题，改拼写即可；3 是设计级反模式——function calling 的本质是"模型只负责**报菜名**（函数名 + JSON 参数），执行永远发生在我们自己的代码里"，而 `eval` 恰好把这道安全边界拆掉了；4 是**能力开关**问题——vLLM 的工具解析默认关闭，客户端写得再对也过不了服务端校验；5 是**模型决策**问题——链路全通了，轮到模型自己决定调不调、调哪个，而 0.6B 在这一步最容易掉链子；6 是**引擎能力边界**问题——API 层接受了参数，引擎层却没人执行它，静默失效比报错更危险。

---

## 二、重点分析：`eval(func_name)(**args)` 为什么必须换掉

### 2.1 机制：eval 是"执行代码"，不是"查函数"

```python
result = eval(func_name)(**args)   # 修复前
```

`eval` 接收**任意字符串**作为 Python 表达式求值。而 `func_name` 来自：

```python
func_name = call.function.name     # 模型生成的文本，经过 vLLM 解析后原样透传
```

也就是说 `eval` 的输入是**模型输出的字符串**。0.6B 级别的小模型在工具调用上偶发幻觉（编造不存在的函数名、复制了描述文本、甚至拼进用户输入的片段），这些字符串会被真的当代码执行。

### 2.2 两层风险

1. **可靠性层（必现）**：模型幻觉出 `get_current_weather`（真实定义叫 `get_current_temperature`）→ `NameError`，脚本当场崩溃。这类崩溃在多轮 agent 循环里尤其难排查，因为出错点距离"模型说错话"隔了一整层解析。
2. **安全层（低概率、高危害）**：`func_name` 若混入可求值的表达式（如被注入的输入），`eval` 会执行任意代码。学习脚本里是假函数无所谓，但这个写法一旦带进生产就是命令注入级别的洞。

### 2.3 修复：映射表 + 查表防御

**第一步，在函数定义之后、TOOLS schema 之前登记映射表**（`test_fcall_basic.py:20-25`）：

```python
# 函数名 → 函数对象 的映射表（生产写法：查表代替 eval，模型返回什么名字都可控）
AVAILABLE_FUNCTIONS = {
    "get_current_temperature": get_current_temperature,
    "get_temperature_date": get_temperature_date,
}
```

**第二步，循环内查表调用**（`test_fcall_basic.py:87-97`）：

```python
for call in tool_calls:
    func_name = call.function.name
    args = json.loads(call.function.arguments)
    func = AVAILABLE_FUNCTIONS.get(func_name)
    if func is None:
        # eval 写法下，模型幻觉出的任意函数名都会被真的执行；查表则只是查不到
        print(f"未知函数 {func_name}，跳过")
        continue
    result = func(**args)
    print(f"调用函数 {func_name}，参数：{args}，结果：{result}")
```

### 2.4 修复前后行为对照

| 模型返回的 `func_name` | `eval` 写法 | 查表写法 |
|---|---|---|
| `get_current_temperature` | 正常执行 | 正常执行 |
| `get_current_weather`（幻觉） | `NameError` 崩溃 | 打印"未知函数"，跳过，循环继续 |
| 任何恶意/畸形字符串 | 当代码执行 | 查不到，跳过 |

查表写法同时是 OpenAI 官方 cookbook 的标准结构：**TOOLS schema 里登记过的名字，才允许被调用**——schema 与映射表天然构成白名单闭环（新增工具时两处各加一条）。

> 学习要点：`eval`/`exec` 处理外部输入（LLM 输出也算外部输入）永远是不达标写法；"动态分发"的正确形态是字典映射或注册表模式。

---

## 三、问题 1、2 的补充说明

**`tool_call` → `tool_calls`（问题 2）**：openai SDK 的 `ChatCompletionMessage` 属性名是复数 `tool_calls`（一次响应可能并行返回多个调用，所以是列表）。写错属性名在 Python 里是运行期才炸的 `AttributeError`——IDE 静态检查能提前抓出来，学习阶段建议开着类型提示写。

**`sysstem` → `system`（问题 1）**：这类拼写错误比崩溃更隐蔽——vLLM 对无法识别的 role 不报错，只是**静默丢弃**该消息的语义，表现为"系统提示没生效但一切看起来正常"。调 prompt 不生效时先查 role 拼写。

---

## 四、二跑问题 4：400 —— 服务端未开启工具解析（重点）

### 4.1 报错现场

客户端脚本已改指向基座服务（`model = "qwen3-0.6b"`），用 `test_vllm_qwen3-0.6b.py` 起服务后跑 `test_fcall_basic.py`：

```
openai.BadRequestError: Error code: 400 - {'error': {'message': '"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set', 'type': 'BadRequestError', 'param': None, 'code': 400}}
```

注意：**400 而不是 404**——说明 `model` 名匹配上了、服务在跑，请求是被 vLLM 的 OpenAI 前端**主动拒绝**的，根本没到模型。

### 4.2 因果链：客户端没写 "auto"，报错却说 "auto"

完整链条分四步，全部有源码依据（vLLM 0.10.2，`ai-gpu` 环境 site-packages）：

**① 客户端只传了 `tools`，没传 `tool_choice`**（`test_fcall_basic.py:69-80`）。这符合 OpenAI API 惯例——"传了 tools 就默认让模型自主决定用不用"。

**② vLLM 协议层自动把 `tool_choice` 补成 `"auto"`**。`vllm/entrypoints/openai/protocol.py:873-876`：

```python
# if "tool_choice" is not specified but tools are provided,
# default to "auto" tool_choice
if "tool_choice" not in data and data.get("tools"):
    data["tool_choice"] = "auto"
```

所以报错信息里的 `'"auto" tool choice'` 不是客户端写的，是 pydantic 校验器补的默认值。

**③ `chat/completions` 处理函数有一道显式校验**。`vllm/entrypoints/openai/serving_chat.py:205-211`：

```python
if (request.tool_choice == "auto" and
        not (self.enable_auto_tools and tool_parser is not None)
        and not isinstance(tokenizer, MistralTokenizer)
        and not self.use_harmony):
    # for hf tokenizers, "auto" tools requires
    # --enable-auto-tool-choice and --tool-call-parser
    return self.create_error_response(
        "\"auto\" tool choice requires "
        "--enable-auto-tool-choice and --tool-call-parser to be set"
    )
```

`self.enable_auto_tools` 与 `tool_parser` 分别来自启动参数 `--enable-auto-tool-choice` 与 `--tool-call-parser`，两者缺一即 400。

**④ 服务端脚本两个参数都没传**。`test_vllm_qwen3-0.6b.py:21-29` 的 `parse_args` 列表里没有任何工具相关参数（基座 / lora / SFT 三个服务端脚本当前都没有），于是 ①→②→③→400 成立。

### 4.3 为什么 vLLM 把它做成"默认关闭"

不同模型家族输出工具调用的**文本格式完全不同**：Qwen 用 `<tool_call>...</tool_call>` 特殊标记包裹 JSON，Llama3 用自己的 JSON 方言，Mistral 靠专用 token，Qwen3-Coder 又是另一套。vLLM 无法从模型目录自动推断该用哪个解析器，只能要求部署者**显式声明**：

- `--enable-auto-tool-choice`：总开关——允许 `tool_choice=auto/required` 路径（即允许 vLLM 替你解析模型输出里的工具调用）；
- `--tool-call-parser <name>`：指定按哪个模型家族的格式去解析。

这是安全设计而非缺陷：解析器配错时轻则工具调用解析失败，重则把模型闲聊文本误判成函数调用。

### 4.4 修复方法（手动改，勿改引擎逻辑）

**只改服务端，客户端不动。** 在 `test_vllm_qwen3-0.6b.py:21-29` 的 `parse_args` 参数列表里加两行（位置随意，建议跟在 `--disable-log-stats` 之后）：

```python
    args = parser.parse_args([
        "--model", main_path+model,
        "--served-model-name", model_name,
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.7",
        "--host", host,
        "--port", port,
        "--disable-log-stats",
        "--enable-auto-tool-choice",          # ← 新增：允许 auto 工具选择
        "--tool-call-parser", "hermes",       # ← 新增：Qwen2.5/Qwen3 系列 parser
    ])
```

要点：

1. **`hermes` 是 Qwen2.5 / Qwen3（含本机 0.6B 及其 LoRA / SFT 变体）的正确选择**。vLLM 0.10.2 实际支持的 parser 可查 `ToolParserManager.tool_parsers`，本机共 20 个，常用的：`hermes`（Qwen2.5/Qwen3）、`qwen3_coder`（仅 Qwen3-Coder 系列）、`llama3_json`、`mistral`、`pythonic`、`glm45`、`deepseek_v3` 等。**不要**因为模型叫 Qwen3 就选 `qwen3_coder`——那是给 Coder 版的，格式不通用。
2. **`--enable-auto-tool-choice` 是开关型参数**，不需要跟值；`--tool-call-parser` 必须跟值。
3. **lora / SFT 两个服务端脚本同理**——哪个脚本要跑 function calling，就给哪个加（LoRA / SFT 微调不改变底层的 hermes 工具标记格式，基座用什么 parser 适配器就用什么）。
4. **改完必须重启服务才生效**（按仓库约定：Ctrl+C → `nvidia-smi` 确认显存归零 → 再起）。启动日志开头会回显解析后的配置，可 grep `enable_auto_tool_choice` 确认为 True。

### 4.5 对 6GB 约束的影响：零

这两个参数是**纯前端（API 层）解析开关**：只影响 HTTP 层如何渲染 prompt、如何解析模型输出文本，不新增 KV cache、不加载额外权重，显存占用与引擎行为完全不变。`--max-model-len 4096`、`VLLM_USE_V1=0`、单引擎等硬性约束全部维持原样，不需要动 OOM 应急三档。

### 4.6 进阶提示：开思维链时的 reasoning parser

当前客户端用 `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` 关闭了思维链，所以不需要额外配置。但如果以后想**保留思维链**做 function calling，服务端还需加 `--reasoning-parser qwen3`（0.10.2 已内置，`vllm/reasoning/qwen3_reasoning_parser.py`）——否则模型输出的 `<think>...</think>` 块会混进 content，干扰 hermes parser 对 `<tool_call>` 标记的定位。

---

## 五、三跑问题 5：模型不调用工具，`tool_calls=[]`（重点）

### 5.1 现象

问题 4 修复后（服务端已带 `--enable-auto-tool-choice --tool-call-parser hermes`），400 消失、脚本跑通，但返回的是纯文本、`tool_calls` 为空列表：

```
ChatCompletionMessage(content='今天深圳的天气情况需要具体的时间来查询。请提供日期以便我为您获取更准确的信息。',
                      ..., tool_calls=[], ...)
```

模型没有发起任何工具调用，反而反问用户要日期。

### 5.2 先排除一个嫌疑：工具定义到底进没进 prompt？（进了）

怀疑链第一环应该是"模型是不是根本没看到工具"。用模型目录里的 tokenizer **离线渲染** chat template（纯 CPU，与服务进程无关），即可复现 vLLM 发给模型的完整 prompt：

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/home/dupengair/shared/LLM/Fine-tuning/model/Qwen3-0.6B")
rendered = tok.apply_chat_template(
    [{"role": "user", "content": "今天深圳天气？"}],
    tools=TOOLS, add_generation_prompt=True, tokenize=False,
    chat_template_kwargs={"enable_thinking": False})
print(rendered)
```

实际输出的 system 段（节选）：

```
<|im_start|>system
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "get_current_temperature", "description": "获取当前城市的温度", ...}}
{"type": "function", "function": {"name": "get_temperature_date", "description": "获取指定日期的温度", ...}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
今天深圳天气？<|im_end|>
<|im_start|>assistant
```

两个函数签名完整注入，输出格式说明也在——**模板链路没问题，工具"送达"了，是模型"拒绝"调用**。这个离线渲染手法本身值得记住：prompt 侧问题不用碰服务，tokenizer 一行就能复现现场。

### 5.3 真正的原因：小模型能力 + 工具描述区分度不足 + 采样偏热

细读那句回复——"需要具体的**时间**来查询"——暴露的是一次**工具前置条件的混淆**：模型把 `get_temperature_date` 的 `date` 必填项错误地套在了"今天深圳天气"这个问题上，而没有意识到 `get_current_temperature` 只要求 `location`、天然适配"今天"。三层因素叠加：

1. **0.6B 的能力天花板（主因）**。多工具场景下"选哪个"是一次分类决策，社区对 Qwen3 小模型的实测普遍结论是：4B 以下工具调用不可靠、易失效易幻觉，稳定 agent 场景建议 8B+（见 5.6 参考）。Qwen3 技术报告也说明小模型的 agent 能力主要靠从大模型蒸馏获得，0.6B 是全系列最小档。
2. **两个工具太相似，description 没有划清边界**。"获取当前城市的温度" vs "获取指定日期的温度"——对人一眼可辨，对 0.6B 不够。模型选工具**几乎完全依赖 description 文本**，而"今天"与"当前"的映射、"指定日期"与"今天"的互斥，它没能自己推出来。参数表里 `date` 必填这个关键差异，只藏在 `required` 数组里，description 里没说。
3. **采样温度偏高**。`temperature=0.7, top_p=0.8` 是 Qwen 官方对**非思考闲聊模式**的推荐值；但工具选择是分类决策，需要确定性，低温（0~0.2）显著更稳。

> 学习要点：function calling 的三段责任——**服务端**负责解析开关（问题 4）、**模型**负责决策调用哪个（本问题）、**客户端**负责真正执行（问题 3 的映射表）。三段全通才是一次成功的工具调用。

### 5.4 修复方法（全部改客户端 `test_fcall_basic.py`，服务端不动）

按性价比排序，建议逐项加、逐项观察：

**修复 1（首选）：把两个工具的 description 写出互斥的适用条件**

```python
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_temperature",
            # 关键：写清"什么时候用我"，并主动排除 date 类问题
            "description": "获取某城市【现在/今天】的实时温度。用户问'今天'、'现在'天气时用这个，本函数不需要日期参数。",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "城市名，如'深圳'"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_temperature_date",
            "description": "获取某城市【指定日期】的温度。仅当用户明确提出具体日期（如'昨天'、'9月15日'、'2026-09-15'）时才用这个；问'今天/现在'请改用 get_current_temperature。",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "城市名，如'深圳'"},
                    "date": {"type": "string", "description": "目标日期，格式 YYYY-MM-DD"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location", "date"]
            }
        }
    }
]
```

写法要点：每个 description 都同时包含**正向条件**（何时用我）与**反向排除**（何时用别人）；`date` 参数补 format 提示。这是小模型场景下提升工具选择准确率最有效的单点改动。

**修复 2：加 system 消息立规矩**

```python
messages = [
    {"role": "system", "content": "你是天气查询助手。回答任何天气问题都必须调用工具获取真实数据，"
                                  "禁止凭空编造，也禁止反问用户。用户没提日期就视为查询当前天气。"},
    {"role": "user", "content": "今天深圳天气？"}
]
```

**修复 3：工具调用场景降温**

```python
temperature=0.0,   # 或 0.1；分类决策要确定性，0.7 是闲聊推荐值
```

**修复 4（~~兜底~~ 本机不可用）：`tool_choice="required"` 强制调用**

```python
response = client.chat.completions.create(
    model="qwen3-0.6b",
    ...,
    tools=TOOLS,
    tool_choice="required",   # 强制模型必须二选一发起调用
    ...
)
```

> ⚠️ **2026-09-17 四跑修正**：本节初稿写"vLLM 0.10.2 支持 required"——这在 API 层面成立（`serving_chat.py:523` 起确有专门分支），但**在本机的 V0 引擎上实际不可用**：required 依赖的 guided decoding / 结构化输出约束在 0.10.2 的 V0 已无实现、参数被静默忽略，模型无约束输出 `<tool_call>` 标签后被 required 解析器按纯 JSON 解析而 400。完整因果链与证据见第六节（问题 6）。教训记入坑 26：**API 接受 ≠ 引擎执行，兜底手段必须在自己的引擎栈上实测**。

**修复 5（学习实验）：单工具基线**

临时注释掉 `get_temperature_date` 相关的 schema、函数与映射表条目，只留 `get_current_temperature` 跑一次。单工具、无混淆场景下 0.6B 大概率能正确调用——先确认最简链路通，再把第二个工具加回去，观察修复 1 的描述改写到底起了多大作用。控制变量，一次只改一项。

### 5.5 预期管理：0.6B 的天花板

即使 1+2+3 全改，0.6B 仍可能偶发不调用/选错/参数填错——这是**模型能力上限，不是代码 bug**。判断标准：偶发失败属正常；若某写法下**稳定**失败，才回头查 schema/参数。想要稳定的 function calling，路径是：换更大模型（8B+），或对小模型做含工具调用样本的微调（本仓库的 SFT/Lora 变体正好可以拿来对比实验——起对应服务端、改客户端 `model` 字段即可）。

修复成功后应看到：

```
ChatCompletionMessage(content=None, ..., tool_calls=[
    ChatCompletionMessageToolCall(id='chatcmpl-tool-...', type='function',
        function=Function(name='get_current_temperature', arguments='{"location": "深圳"}'))])
调用函数 get_current_temperature，参数：{'location': '深圳'}，结果：{'temperature': 26.1, ...}
```

注意成功时 `content` 通常为 `None`、工具调用走 `tool_calls` 字段——脚本的解析循环正是读这个字段，无需再改。

### 5.6 参考来源

- [Qwen3 小模型 MCP 实测（4B~30B，工具调用失效/幻觉问题）](https://blog.csdn.net/GeoSmart/article/details/147676866)
- [Luke Fan：阿里 Qwen3 评测（4B 及以下工具/代码调用易出错）](https://lukefan.com/2025/05/04/)
- [Qwen3 技术报告（arXiv，小模型 agent 能力蒸馏）](https://arxiv.org/html/2505.09388v1)
- [Qwen3 系列规格（小模型仅 32K 上下文）](https://ginonotes.com/posts/qwen3-released)

---

---

## 六、四跑问题 6：`tool_choice="required"` 在 V0 引擎上约束静默失效（重点）

### 6.1 现象

应用修复 1（描述区分度）+ 修复 2（system 立规）+ 修复 4（`tool_choice="required"`）后运行：

```
openai.BadRequestError: Error code: 400 - {'error': {'message': '1 validation error for list[function-wrap[__log_extra_fields__()]]\n
  Invalid JSON: expected value at line 1 column 1 [type=json_invalid,
  input_value=\'<tool_call>\\n{"name": "g...深圳"}}\\n</tool_call>\', input_type=str]\n
  For further information visit https://errors.pydantic.dev/2.12/v/json_invalid', ...}}
```

### 6.2 第一步：读懂报错里的三个关键信息

这个报错看着吓人，其实信息全在明面上：

1. **`input_value='<tool_call>\n{"name": "g...深圳"}}\n</tool_call>'`（中间被 pydantic 截断）——这是模型的输出**，不是请求参数。展开即 `<tool_call>\n{"name": "get_current_temperature", "arguments": {"location": "深圳"}}\n</tool_call>`。两个立刻可下的结论：
   - **修复 1+2 已经生效**：模型选对了函数（`get_current_temperature`，不是 date 版——截断尾部是"深圳"而不是某个日期，说明参数只有一个 location），参数也对。0.6B 完成了正确的"报菜名"。
   - 模型输出的是 **Qwen3 chat template 规定的 hermes 格式**（JSON 外面包 `<tool_call>` 标签，见 5.2 节渲染的 system 指令），不是纯 JSON。
2. **`Invalid JSON: expected value at line 1 column 1`**——JSON 解析在第 1 行第 1 列就失败，因为第一个字符是 `<`，不是 `{` 或 `[`。说明有个环节把"整段模型输出"当**纯 JSON** 解析了。
3. **`list[function-wrap[__log_extra_fields__()]]`**——被校验的目标类型是"某个模型类的列表"，且该模型类带一个叫 `__log_extra_fields` 的包装校验器。到 `protocol.py:66` 一查：`OpenAIBaseModel` 的 `@model_validator(mode="wrap")` 正是这个名字，`FunctionDefinition` 继承自它。即校验类型是 `list[FunctionDefinition]`。

### 6.3 因果链：required 的约束为什么没拦住模型

**`tool_choice="required"` 的设计意图**：不靠模型自觉，而是用 **guided decoding（结构化输出）** 在解码时用语法直接限制输出只能是符合 tools schema 的 JSON 数组。链路本应是：客户端声明 required → 服务端构建 JSON schema → 引擎在解码时逐 token 约束 → 模型只能输出 `[{"name": ..., "parameters": ...}]` → 服务端 `validate_json` 解析。本机实际发生的是（全部有源码依据）：

**① 协议层确实构建了约束 schema**。`protocol.py` 的 `_get_guided_json_from_tool()`（738 行起）：`tool_choice == "required"` 分支用两个工具的 name enum + parameters 拼出 `{"type": "array", "minItems": 1, "items": {...}}` 的 JSON schema，经 `GuidedDecodingParams.from_optional(...)`（690 行）塞进 `SamplingParams.guided_decoding`（731 行）传给引擎。

**② V0 引擎根本没人消费这个参数（本问题的根因）**。在 0.10.2 的安装目录全量 grep `guided_decoding`：V0 侧只剩 `arg_utils.py`（CLI 参数定义）和 `transformers_utils/tokenizers/mistral.py`（Mistral 专用路径）两处引用；`engine/async_llm_engine.py` 等 V0 执行链路**零引用**。而 V1 侧有完整实现（`v1/core/sched/scheduler.py` 等）。结论：**0.10.2 的 V0 引擎已不再实现 guided decoding，`guided_decoding` 参数被静默忽略**——V0 处于维护模式，结构化输出是 V1 专属特性。本机 `VLLM_USE_V1=0` 是硬性约束（6GB + WSL2，V1 未验证），正好踩中这个缺口。

**③ 模型于是无约束地按模板指令输出**。chat template 的 system 段明确要求"在 `<tool_call></tool_call>` XML 标签内返回 JSON"（5.2 节渲染原文），模型忠实执行——输出对 hermes 格式来说**完全正确**，但对 required 解析器来说是"脏"的。

**④ required 分支的解析器只认纯 JSON**。`serving_chat.py:1288`：

```python
elif request.tool_choice and request.tool_choice == "required":
    ...
    # the fields of FunctionDefinition are a superset of the
    # tool call outputs and can be used for parsing
    assert content is not None
    tool_calls = TypeAdapter(
        list[FunctionDefinition]).validate_json(content)   # ← 把整段输出当纯 JSON
```

`<tool_call>...` 第一个字符 `<` 不是合法 JSON 起始 → pydantic `json_invalid` → 400。**注意它不做剥离标签的兜底**（auto 路径才会走 hermes parser）。

> 一句话总结因果：**API 层收下了 required、协议层造好了语法、引擎层（V0）没有执行、解析层只认纯 JSON——中间一环断裂，两头都对，错在链路不通。**

### 6.4 修复方法（改客户端 `test_fcall_basic.py`，服务端不动）

**删掉第 76 行 `tool_choice="required",`（回到 auto 路径）即可**：

```python
response = client.chat.completions.create(
        model = "qwen3-0.6b",
        messages = messages,
        tools=TOOLS,
        # tool_choice 不写，协议层自动补 "auto"（见坑 24）
        temperature=0.7,
        ...
)
```

为什么这样就对了：auto 路径走的是**已配置的 `--tool-call-parser hermes`**（问题 4 加的），hermes parser 的本职工作就是解析 `<tool_call>{json}</tool_call>`——6.2 节的报错恰好证明模型这次输出的正是标准 hermes 格式。等于说：**模型已经把菜名报对了，之前只是被 required 抢走了传菜员的活**。

预期成功输出：

```
ChatCompletionMessage(content='', ..., tool_calls=[
    ChatCompletionMessageToolCall(id='chatcmpl-tool-xxxx', type='function',
        function=Function(name='get_current_temperature',
                          arguments='{"location": "深圳"}'))], ...)
调用函数 get_current_temperature，参数：{'location': '深圳'}，结果：{'temperature': 26.1, 'location': '深圳', 'unit': 'celsius'}
```

补充两点：

1. `temperature` 你保留了 0.7——这次在 0.7 下模型已能正确发起调用，可不改；若多跑几次出现偶发不调用/选错，再按修复 3 降到 0.0~0.2。
2. **不建议为了用 required 而切 V1**（`VLLM_USE_V1=1`）：本仓库约定 V1 在此机器未验证，且 6GB 约束下的排错成本高。auto + hermes 已是本机正确组合。

### 6.5 V0/V1 能力差异备忘（0.10.2，与本机相关项）

| 能力 | V0（本机） | V1 |
|---|---|---|
| `--enable-auto-tool-choice` + `--tool-call-parser`（auto 路径，服务端解析） | ✅ | ✅ |
| `tool_choice="required"` / `guided_json` 等结构化输出约束 | ❌ 参数被静默忽略 | ✅ 完整实现 |
| `--reasoning-parser`（思维链剥离） | ✅ | ✅ |

> 学习要点：报错时先找 `input_value` / `input_type` 这类**实际数据**字段——它直接告诉你"什么东西在哪个环节被谁处理失败了"；再顺着被校验的类型名（`__log_extra_fields__`）反查源码定位校验点。比盯着错误码猜快得多。另外，"API 接受了参数"只说明协议层通过，**参数最终由引擎执行，引擎不支持的特性会静默失效**——兜底方案必须在自己的引擎栈上实测（坑 26）。

---

## 七、复跑核对清单

- [ ] 先起对应服务端：`python test_vllm_qwen3-0.6b.py`（脚本请求的 model 是 `qwen3-0.6b`，起错变体会 404；要测 lora/sft 版则起对应脚本并同步改客户端 `model` 字段）；
- [ ] 确认服务端脚本已加 `--enable-auto-tool-choice` + `--tool-call-parser hermes`（问题 4 的修复），启动日志里 `enable_auto_tool_choice=True`；
- [ ] `curl http://localhost:8000/v1/models` 确认返回 `qwen3-0.6b`；
- [ ] 确认客户端已删掉 `tool_choice="required"`（问题 6 的修复，回到 auto + hermes）；
- [ ] 跑 `python test_fcall_basic.py`，观察 `tool_calls` 从 `[]` 变为含一条 `get_current_temperature` 调用（修复 1+2 已让模型学会报菜名，报错那次的输出就是证明）；
- [ ] 观察第一段打印的 `message` 里 `tool_calls` 结构（`id` / `function.name` / `function.arguments` 是 JSON 字符串，需 `json.loads`）;
- [ ] 确认输出 `调用函数 get_current_temperature，参数：{'location': '深圳'}，结果：{'temperature': 26.1, ...}`；
- [ ] （实验）换 `get_temperature_date` 场景问"深圳 9月15日 温度？"，验证两个工具的路由都正确；
- [ ] （进阶）把问题 1、2 的错误写法临时改回去复现一次崩溃，加深印象后还原。

## 八、踩坑归档

**坑 22：function calling 的执行侧必须用白名单映射表，禁止 `eval` 模型输出的函数名**。模型只产出"函数名 + JSON 参数"文本；`call.function.arguments` 是字符串要 `json.loads`；`call.function.name` 是模型幻觉高发区，`.get()` 查不到就跳过。schema（给模型看的）与映射表（给自己用的）成对维护。

**坑 23：SDK 属性名错误是运行期错误，role 拼写错误是静默失效**——前者靠 `AttributeError` 现场报错定位，后者表现为"功能没生效但不报错"。vLLM/OpenAI 兼容层对畸形字段普遍采取宽容丢弃策略，调试时要主动校验请求结构。

**坑 24：vLLM 的 tools 能力默认关闭，客户端"只传 `tools` 不传 `tool_choice`"会走 auto 路径**。协议层（protocol.py 校验器）自动补 `tool_choice="auto"`，而 auto 路径要求服务端显式给 `--enable-auto-tool-choice` + `--tool-call-parser <name>`，缺一即 400（serving_chat.py 校验，请求进不到模型）。Qwen2.5/Qwen3 系（含 LoRA/SFT 变体）用 `hermes`；`qwen3_coder` 只给 Qwen3-Coder。这两个是纯前端解析参数，不占显存，与 6GB 约束无关；改后须重启服务。若保留思维链，另需 `--reasoning-parser qwen3`。

**坑 25：链路通了 ≠ 模型会调——`tool_calls=[]` 先查 description 区分度，再怀疑模型**。工具定义是否进 prompt 可用 `tokenizer.apply_chat_template(..., tools=...)` 离线渲染验证（不碰服务）；确认送达后，原因基本是模型侧决策失败。小模型（0.6B 级）选工具全靠 description 文本：相似工具必须写互斥的正反条件（何时用我/何时用别人），关键参数差异（如 date 必填）要写进 description 而不是只藏在 required 数组里；采样要降温（0~0.2，0.7 是闲聊值）。成功调用的返回特征：`content=None`、信息全在 `tool_calls` 列表里。

**坑 26：`tool_choice="required"` / guided decoding 等结构化输出在 vLLM 0.10.2 的 V0 引擎上已被静默移除，本机（`VLLM_USE_V1=0`）禁用**。API 层照常接受参数、协议层照常构建 JSON schema（`_get_guided_json_from_tool`），但 V0 执行链路无人消费 `SamplingParams.guided_decoding`（全量 grep 可证，V1 才有实现）——模型无约束输出 hermes 格式 `<tool_call>{json}</tool_call>`，required 解析器（`serving_chat.py:1288` 的 `validate_json`）按纯 JSON 解析在第一个 `<` 上炸掉，报 pydantic `json_invalid` 400。本机正确组合永远是 **auto + `--tool-call-parser hermes`**。通用教训：**API 接受 ≠ 引擎执行**——查一个特性是否可用，不能只看请求过没过协议层校验，要查引擎侧有没有消费代码；报错时优先读 `input_value` 字段还原"什么东西被谁处理失败"。
