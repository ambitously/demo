from openai import OpenAI


client = OpenAI(
    api_key="sk-80deba4ff33c44cd82b0dd29bf43a6fb",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def main():
    response = client.chat.completions.create(
    model="qwen3-8b",
    messages=[
        {"role": "user", "content": "hello"}
    ],
    extra_body={"enable_thinking": False},
)
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
