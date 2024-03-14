import base64
import os

import gradio as gr
import requests

from dotenv import load_dotenv

load_dotenv()


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def translate(image):
    api_key = os.environ.get("OPENAI_API_KEY")
    base64_image = encode_image(image)

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "system",
                        "text": """
                            You will receive a screenshot of an anonymized text conversation. Analyze the
                            screenshot and provide a summary of the conversation.
                        """,
                    }
                ],
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ],
            }
        ],
        "max_tokens": 300,
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )
    response = response.json()

    if "choices" in response and len(response["choices"]) > 0:
        response = response["choices"][0]["message"]["content"]
    else:
        return "No response found."

    followup = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "user",
                "content": """
                    You are a dating coach, helping clients understand the meaning of their text conversations.
                    they have. When a client sends you a description of their text conversation, analyze it
                    and provide advice on how to respond to the conversation.
                """,
            },
            {
                "role": "user",
                "content": f"Here is the conversation: {response}",
            },
            {"role": "user", "content": "What advice would you give?"},
        ],
        "max_tokens": 300,
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=followup
    )
    response = response.json()

    if "choices" in response and len(response["choices"]) > 0:
        return response["choices"][0]["message"]["content"]
    else:
        return "No response found."


demo = gr.Interface(
    fn=translate,
    inputs=gr.Image(type="filepath"),
    outputs=["text"],
    title="groupchat",
    description="Upload your screenshot on the left and we'll give you advice on the right."
)

demo.launch(share=True)
