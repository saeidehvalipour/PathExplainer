import openai
from src.config import get_openai_api_key

api_key = get_openai_api_key()
openai.api_key = api_key

def oai_get_response(msg, model="gpt-4", temp=0.5, top_p=1.0, seed=None):

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "user", "content": msg}
        ],
        temperature=temp,
        top_p=top_p
    )
    return response.choices[0].message.content
