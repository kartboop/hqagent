import hqagent

# 方式1: 直接传参
# client = hqagent.ChatClient("https://api.openai.com/v1", "sk-xxx", "gpt-4")

# 方式2: 从 .env 读取 (BASE_URL, API_KEY, MODEL_NAME)
client = hqagent.ChatClient.from_env()

# 非流式聊天
# resp = client.chat([{"role": "user", "content": "Hello"}])
# print(resp)  # dict: {id, choices, usage, ...}

# 流式聊天 (收集所有 chunk)
chunks = client.chat_stream([{"role": "user", "content": "Hello"}])
for chunk in chunks:
    print(chunk)  # 每个 chunk 是 dict