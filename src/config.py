# src/config.py

def get_openai_api_key():
    with open("OAI_CONFIG_LIST", "r") as file:
        return file.read().strip()
