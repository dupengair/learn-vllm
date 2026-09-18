from openai import OpenAI

base_url = "http://localhost:8000/v1"
client = OpenAI(api_key="EMPTY", base_url=base_url)

#model = "qwen3-0.6b-lora"
#model = "qwen3-0.6b-sft"
model = "qwen3-0.6b"

response = client.chat.completions.create(
        model = model,
        messages = [
            {"role":"system","content":"假设你是一名粤菜大厨"},
            {"role":"user","content":"东莞有哪些美食"}
        ]
    )
print(response.choices[0].message)