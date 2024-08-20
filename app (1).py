import torch
from llama_index.core.prompts import PromptTemplate
from transformers import AutoTokenizer
from llama_index.core import Settings
import os
import time
from llama_index.llms.text_generation_inference import TextGenerationInference
import whisper
import gradio as gr
from gtts import gTTS

model = whisper.load_model("base")
HF_API_TOKEN = os.getenv("HF_TOKEN")

def translate_audio(audio):

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # decode the audio
    options = whisper.DecodingOptions(language='en', task="transcribe", temperature=0)
    result = whisper.decode(model, mel, options)
    return result.text

def audio_response(t):
    tts = gTTS(text=t, lang='en', slow=False)
    tts.save("output.mp3")
    mp3_file_path = "output.mp3"
    return mp3_file_path

def messages_to_prompt(messages):
    # Default system message for a chatbot
    default_system_prompt = "You are an AI chatbot designed to assist with user queries in a friendly and conversational manner."

    prompt = default_system_prompt + "\n"

    for message in messages:
        if message.role == 'system':
            prompt += f"\n{message.content}</s>\n"
        elif message.role == 'user':
            prompt += f"\n{message.content}</s>\n"
        elif message.role == 'assistant':
            prompt += f"\n{message.content}</s>\n"

    # Ensure we start with a system prompt, insert blank if needed
    if not prompt.startswith("\n"):
        prompt = "\n</s>\n" + prompt

    # Add final assistant prompt
    prompt = prompt + "\n"

    return prompt

def completion_to_prompt(completion):
    return f"<|system|>\n</s>\n<|user|>\n{completion}</s>\n<|assistant|>\n"

Settings.llm = TextGenerationInference(
    model_url="https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct",
    token=HF_API_TOKEN,
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt
)
def text_response(t):
    time.sleep(1)  # Adjust the delay as needed
    response = Settings.llm.complete(t)
    message = response.text
    return  message

def transcribe_(a):
    t1 = translate_audio(a)
    t2 = text_response(t1)
    t3 = audio_response(t2)
    return (t1, t2, t3)

output_1 = gr.Textbox(label="Speech to Text")
output_2 = gr.Textbox(label="LLM Output")
output_3 = gr.Audio(label="LLM output to audio")

gr.Interface(
    title='AI Voice Assistant',
    fn=transcribe_,
    inputs=[
        gr.Audio(sources="microphone", type="filepath"),
    ],
    outputs=[
        output_1, output_2, output_3
    ]
).launch(share=True)

