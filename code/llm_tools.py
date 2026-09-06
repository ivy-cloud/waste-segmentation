import os
from openai import OpenAI


class UseLLM():
    def __init__(self, api_key, model="gpt-4o-mini"):
        self.model = model
        self.client = OpenAI(
            api_key = api_key,
            base_url = None, #https://api.moonshot.cn/v1
        )

    def use(self, message):
        completion = self.client.chat.completions.create(
            model = self.model,
            messages = [
                {"role": "system", "content": "You are an expert in environmental protection, "
                                              "and we need you to respond with content related "
                                              "to waste disposal."},
                {"role": "user", "content": message}
            ],
            temperature = 0.3,
        )
        return completion.choices[0].message.content
    

if __name__ == '__main__':
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("API_KEY is not set.")
    chatgpt = UseLLM(api_key=api_key)
    message = 'paper Is it recyclable waste? If yes, return True; if no, return False.'
    result = chatgpt.use(message=message)
    print(result)
