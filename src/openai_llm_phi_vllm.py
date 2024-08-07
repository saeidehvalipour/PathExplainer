import openai
from openai import OpenAI

def setup_openai_client(api_key_path='OAI_CONFIG_LIST'):
    with open(api_key_path) as f:
        api_key = f.read().strip()
    return openai.OpenAI(api_key=api_key, base_url="http://localhost:8000/v1")

client = setup_openai_client()
llm_model = "microsoft/Phi-3-mini-4k-instruct"

# Function to get response from vllm locally 
def vllm_get_response(prompt, temp=1e-19, top_p=1e-9, seed=1234):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=llm_model,
        temperature=temp,
        top_p=top_p,
    )

    reply_text = chat_completion.choices[0].message.content
    return reply_text