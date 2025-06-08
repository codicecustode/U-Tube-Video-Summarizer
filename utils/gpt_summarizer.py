from openai import OpenAI
import os
client = OpenAI(api_key='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')

response = client.responses.create(
    model="o3",
    input="Write a one-sentence bedtime story about a unicorn."
)

print(response.output_text)