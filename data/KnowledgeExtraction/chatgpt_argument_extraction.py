import openai

openai.api_key = "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"
prompt = "下面是和 AI 助手的对话。这个助手热情、聪明、友好。\n\n人类: 你好，你是谁？\nAI 助手:"

response = openai.Completion.create(
  model="gpt-3.5-turbo",
  prompt=prompt,
  temperature=0.1,
  max_tokens=4096,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0.6,
  stop=[" 人类:", " AI 助手:"]
)

print(response['choices'][0]['text'])