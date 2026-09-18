import json
from openai import OpenAI

# 定义 function
device_state = {
    "living_room": "off",
    "bedroom": "off",
    "kitchen": "off"
}

def control_lights(operations):
    """
    批量控制灯光。
    operations 是一个列表，包含多个操作对象：{"room": "living_room", "state": "on"}
    """
    results = []
    for op in operations:
        room = op.get("room")
        state = op.get("state")
        if room in device_state:
            device_state[room] = state
            results.append(f"{room} 已切换为 {state}")
        else:
            results.append(f"错误：找不到房间 {room}")
    return json.dumps(results)

AVAILABLE_FUNCTIONS = {
    "control_lights": control_lights,
}


# 定义 Tools Schema
# 注意：这里定义了一个包含 array (数组) 的复杂结构
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "control_lights",
            "description": "批量控制家里的灯光开关",
            "parameters": {
                "type": "object",
                "properties": {
                    "operations": {
                        "type": "array",  # 参数类型是数组
                        "description": "操作列表",
                        "items": {
                            "type": "object",
                            "properties": {
                                "room": {
                                    "type": "string",
                                    "enum": ["living_room", "bedroom", "kitchen"], # 限制只能填这三个房间
                                    "description": "房间名称"
                                },
                                "state": {
                                    "type": "string",
                                    "enum": ["on", "off"], # 限制只能填开或关
                                    "description": "目标状态"
                                }
                            },
                            "required": ["room", "state"]
                        }
                    }
                },
                "required": ["operations"]
            }
        }
    }
]

base_url = "http://localhost:8000/v1"
client = OpenAI(api_key="EMPTY", base_url=base_url)

prompt = "我回来了，帮我把客厅和卧室的灯都打开，但是厨房的灯关掉。"
messages = [
    {"role": "system", "content": "你是智能家居管家"},
    {"role": "user", "content": prompt}
]

response = client.chat.completions.create(
        model = "qwen3-0.6b",
        messages = messages,
        tools=tools_schema,
        tool_choice="auto",
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

