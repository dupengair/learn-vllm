import json
from openai import OpenAI

# 定义 function
def get_current_temperature(location: str, unit: str = "celsius"):
    return {
        "temperature": 26.1,
        "location": location,
        "unit": unit,
    }

def get_temperature_date(location: str, date: str, unit: str = "celsius"):
    return {
        "temperature": 25.9,
        "location": location,
        "date": date,
        "unit": unit,
    }

# 函数名 → 函数对象 的映射表（生产写法：查表代替 eval，模型返回什么名字都可控）
AVAILABLE_FUNCTIONS = {
    "get_current_temperature": get_current_temperature,
    "get_temperature_date": get_temperature_date,
}

# 定义 Tools Schema
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


base_url = "http://localhost:8000/v1"
client = OpenAI(api_key="EMPTY", base_url=base_url)

messages = [
    {"role": "system", "content": "你是天气查询助手。回答任何天气问题都必须调用工具获取真实数据，"
                                  "禁止凭空编造，也禁止反问用户。用户没提日期就视为查询当前天气。"},
    {"role": "user", "content": "今天深圳天气？"}
]

response = client.chat.completions.create(
        model = "qwen3-0.6b",
        messages = messages,
        tools=TOOLS,
        temperature=0.7,
        top_p=0.8,
        max_tokens=512,
        extra_body={
            "repetition_penalty": 1.05,
            "chat_template_kwargs": {"enable_thinking": False}
        }
    )

print(response.choices[0].message)

# 解析工具调用
tool_calls = response.choices[0].message.tool_calls
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

