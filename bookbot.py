import openai
import pandas as pd
import time
import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials

#security keys removed because of security reasons of github
openai_api_key = ""
openai_endpoint = "https://book-openai.openai.azure.com/"
vision_key = ""
vision_endpoint = "https://book-vision.cognitiveservices.azure.com/"
speech_key = ""
speech_region = "australiaeast"

books_df = pd.read_csv("books.csv")

openai.api_key = openai_api_key
openai.api_base = openai_endpoint
openai.api_type = 'azure'
openai.api_version = '2023-05-15'

def get_openai_response(prompt):
    client = openai.AzureOpenAI(
        api_key=openai.api_key,
        api_version=openai.api_version,
        azure_endpoint=openai.api_base
    )
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "You are a bookshop assistant recommending books to customers based on their preferences."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


def recommend_books(user_input):
    if 'genres' in books_df.columns:
        matches = books_df[
            books_df['title'].str.contains(user_input, case=False, na=False) |
            books_df['authors'].str.contains(user_input, case=False, na=False) |
            books_df['genres'].str.contains(user_input, case=False, na=False)
        ].head(5)
    else:
        matches = books_df[
            books_df['title'].str.contains(user_input, case=False, na=False) |
            books_df['authors'].str.contains(user_input, case=False, na=False)
        ].head(5)

    if matches.empty:
        return "No matching books found."
    
    return "\n".join(
        f"{row['title']} by {row['authors']} (Rating: {row['average_rating']})"
        for _, row in matches.iterrows()
    )


def analyze_image(image_file):
    client = ComputerVisionClient(
        vision_endpoint,
        CognitiveServicesCredentials(vision_key)
    )
    result = client.read_in_stream(image_file, raw=True)
    op_location = result.headers["Operation-Location"]
    op_id = op_location.split("/")[-1]
    while True:
        result = client.get_read_result(op_id)
        if result.status not in ['notStarted', 'running']:
            break
        time.sleep(1)
    text = ""
    if result.status == 'succeeded':
        for line in result.analyze_result.read_results[0].lines:
            text += line.text + " "
    return text.strip()

def recognize_speech():
    config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    recognizer = speechsdk.SpeechRecognizer(speech_config=config, audio_config=audio_config)
    result = recognizer.recognize_once()
    return result.text if result.reason == speechsdk.ResultReason.RecognizedSpeech else "Could not understand."

def synthesize_speech(text):
    config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    audio_config = speechsdk.audio.AudioOutputConfig(filename="static/response.wav")
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=config, audio_config=audio_config)
    synthesizer.speak_text_async(text).get()
