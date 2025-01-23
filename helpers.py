import base64
import io
import os
import re
import validators

import pandas as pd
import streamlit as st

from dotenv import load_dotenv
from groq import Groq
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from PIL import Image
from pytubefix import YouTube
from qdrant_client import QdrantClient

VECTOR_STORE_PATH = os.path.join('.', '.vector_store')
TEMP_PATH = os.path.join('.', '.temp')
TRANSFORMERS_CACHE = os.path.join('.', '.transformers_cache')

# Initialize settings
load_dotenv()

client = Groq(api_key=os.getenv('GROQ_API_KEY'))
client_db = QdrantClient(path=os.path.join('.', '.qdrant_db'))
embed_model = HuggingFaceEmbedding(
    model_name='sentence-transformers/all-MiniLM-L12-v2',
    cache_folder=TRANSFORMERS_CACHE
)


def get_storage_context():
    if not os.path.exists(VECTOR_STORE_PATH):
        vector_store = QdrantVectorStore(
            client=client_db,
            collection_name="rag"
        ) 
        storage_context = StorageContext.from_defaults(vector_store=vector_store)        
        index = VectorStoreIndex.from_documents(
            [],
            embed_model=embed_model,
        )
        index.storage_context.persist(VECTOR_STORE_PATH)
    return StorageContext.from_defaults(persist_dir=VECTOR_STORE_PATH)


@st.cache_resource
def get_whisper_model():
    try:
        return "whisper-large-v3-turbo"
    except Exception as e:
        st.error(f"Error loading transcription model: {e}")
        st.stop()

def transcribe_audio(audio_file_path, show_message=True):
    try:
        model = get_whisper_model()
        if show_message:
            st.write("Transcribing audio file...")
        with open(audio_file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=model,
                file=audio_file
            )
        transcription_text = transcription.text.strip() if hasattr(transcription, 'text') else ""
        return transcription_text
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return ""


def download_youtube_audio(youtube_link):
    pattern = r"^https?:\/\/(?:www\.)?youtube\.com\/(?:watch\?v=)?([^&\s]+)"
    match = re.match(pattern, youtube_link)    
    if match:
        video_id = match.group(1)
        try:
            yt = YouTube(f"https://www.youtube.com/watch?v={video_id}", client='MWEB')
            stream = yt.streams.get_audio_only()
            if stream is None:
                st.error("No audio streams available for this video.")
                return None            
            audio_file = stream.download(output_path=TEMP_PATH)
            return audio_file
        except Exception as e:
            st.error(f"Failed to download YouTube audio: {e}")
            return None
    else:
        st.error("Invalid YouTube link.")
        return None
    

def file_to_df(df_file_path, file_extension):
    if file_extension == '.csv':
        df = pd.read_csv(df_file_path)
    elif file_extension in ['.xls', '.xlsx']:
        df = pd.read_excel(df_file_path)
    else:
        raise ValueError("Unsupported file format: {}".format(file_extension))    
    return df


def vision_model(question, image_content):
    client = Groq()
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    image_content,
                ],
            }
        ],
        model="llama-3.2-90b-vision-preview",
    )
    return chat_completion

def image_to_base64(img_file_path):   
    with Image.open(img_file_path) as img:
        buffered = io.BytesIO()        
        img.save(buffered, format="PNG")        
        img_bytes = buffered.getvalue()        
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')        
    return img_base64


def website_checker(url: str):
    youtube_regex = (
        r'^(https?://)?(www\.)?'
        r'((youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/|youtube\.com/shorts/|youtube\.com/channel/|youtube\.com/.*?v=)|'
        r'(youtube\.com/.*?))([a-zA-Z0-9_-]{11})$'
    )
    if re.match(youtube_regex, url):
        return url, None    
    if validators.url(url):
        return None, url
    else:
        return None, None
