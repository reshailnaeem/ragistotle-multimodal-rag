import inspect
import os
import shutil
import sys

import asyncio
import nest_asyncio

import streamlit as st

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from llama_index.core import (
    Document,
    Settings,
    SimpleDirectoryReader,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from pandasai import Agent
from pandasai import SmartDataframe
from scrapegraphai.graphs import SmartScraperGraph
from scrapegraphai.utils import prettify_exec_info

from helpers import (
    TEMP_PATH,
    TRANSFORMERS_CACHE,
    VECTOR_STORE_PATH,
    get_storage_context,
    transcribe_audio,
    download_youtube_audio,
    file_to_df,
    image_to_base64,
    vision_model,
    website_checker
)

# Initialize settings
load_dotenv()

Settings.llm = Groq(model='llama-3.3-70b-versatile', api_key=os.getenv('GROQ_API_KEY'), max_tokens=1200)
embed_model = HuggingFaceEmbedding(
    model_name='sentence-transformers/all-MiniLM-L12-v2',
    cache_folder=TRANSFORMERS_CACHE
)
llm = ChatGroq(
    model='llama-3.3-70b-versatile',
    api_key=os.getenv('GROQ_API_KEY'),
    max_tokens=800,
)

# Some PandasAI thing to not use their training data
os.environ.pop("PANDASAI_API_KEY", None)

# # Windows event loop compatibility
# asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
# nest_asyncio.apply()

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# Set visual aspects
st.set_page_config(page_title="🤖🏛️📜 RAGistotle")

col1, col2, col3 = st.columns(3)
with col2:
    st.image('header.png', width=250)

st.header("🤖🏛️📜 RAGistotle")

uploaded_file_name = None
document_file = None
audio_file = None
video_file = None
agent = None
base64_image = None
youtube_link = None
website_link = None
question = ''

with st.container():
    st.subheader("You know what to do. 📂")
    uploaded_file = st.file_uploader(
        "Upload file",
        help="Supported formats: text, Word documents, PDF, MP3, WAV, MP4, CSV, Excel sheets, JPG, PNG"
    )
    url_link = st.text_input("Enter your link")

    if url_link:
        youtube_link, website_link = website_checker(url_link)
        if youtube_link:
            st.write("YouTube link detected:", youtube_link)
        elif website_link:
            st.write("Website link detected:", website_link)
        else:
            st.write("Invalid or unsupported link.")
    
    if not uploaded_file and not url_link:
        st.stop()
    
    # Create temp folder if not available
    if not os.path.exists(TEMP_PATH):
        os.mkdir(TEMP_PATH)

    if uploaded_file:
        uploaded_file_name = uploaded_file.name
        uploaded_file_buffer = uploaded_file.getbuffer()
        temp_file_path = os.path.join(TEMP_PATH, uploaded_file_name)

        # Text support
        if uploaded_file_name.endswith(('.txt', '.doc', '.docx', '.pdf')):
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file_buffer)
            
            documents = SimpleDirectoryReader(input_files=[temp_file_path]).load_data()
            document_file = documents[0].text
      
        # Audio support
        elif uploaded_file_name.endswith(('.mp3', '.wav')):
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file_buffer)

            transcription_text = transcribe_audio(temp_file_path)
            audio_file = transcription_text if transcription_text else None
            
        # Video support
        elif uploaded_file_name.endswith('.mp4'):
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file_buffer)

            transcription_text = transcribe_audio(temp_file_path)
            video_file = transcription_text if transcription_text else None
            
        # Data formats support
        elif uploaded_file_name.endswith(('.csv', '.xls', '.xlsx')):
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file_buffer)
            
            file_extension = os.path.splitext(uploaded_file_name)[1]
            df_raw = file_to_df(temp_file_path, file_extension)
            df = SmartDataframe(df_raw)
            agent = Agent([df], config={"llm": llm, "enable_cache": False, "verbose": True})

        # Image support
        elif uploaded_file_name.lower().endswith(('.jpg', '.jpeg', '.bmp', '.png')):
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file_buffer)

            base64_image  = image_to_base64(temp_file_path)
            image_content = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}

        try:
            os.unlink(temp_file_path)
        except OSError:
            st.error("Error deleting temporary file.")

    # YouTube support
    if youtube_link:
        st.write("Downloading audio from YouTube...")
        yt_audio_file = download_youtube_audio(youtube_link)

        if yt_audio_file:
            audio_temp_file = os.path.join(TEMP_PATH, yt_audio_file)
            uploaded_file_name = yt_audio_file

            transcription_text = transcribe_audio(audio_temp_file)
            audio_file = transcription_text if transcription_text else None
            
            os.unlink(audio_temp_file)

    elif website_link:
        graph_config = {
            "llm": {
                "model": "groq/gemma2-9b-it",
                "api_key": os.getenv('GROQ_API_KEY'),
                "temperature": 0.3
            },
            "headless": True,
            "verbose": True,
        }        

        smart_scraper_graph = SmartScraperGraph(
            prompt=question,
            source=website_link,
            config=graph_config
        )

        st.write("Scraping website...")

    # Load index and insert file
    if document_file or audio_file or video_file:
        if document_file:
            processed_file = document_file
        elif audio_file:
            processed_file = audio_file
        elif video_file:
            processed_file = video_file
        index = load_index_from_storage(storage_context=get_storage_context(), embed_model=embed_model)
        index.insert(Document(text=processed_file, extra_info={"file_name": uploaded_file_name or "youtube_audio.mp3"}))
        index.storage_context.persist(VECTOR_STORE_PATH)

        if document_file:
            st.write("Text file inserted and processed successfully. Ready to query!")
        elif audio_file:
            st.write("Audio inserted and processed successfully. Ready to query!")
        elif video_file:
            st.write("Video inserted and processed successfully. Ready to query!")

    elif agent or base64_image:
        index = load_index_from_storage(storage_context=get_storage_context(), embed_model=embed_model)
        index.storage_context.persist(VECTOR_STORE_PATH)
        
        if agent:
            st.write("Data inserted and processed successfully. Ready to query!")
        elif base64_image:
            st.write("Image inserted and processed successfully. Ready to query!")

# Query the data
st.subheader("Ask your question. 🔍")
question = st.text_area("Enter your question...")

if question:
    querying_placeholder = st.empty()
    querying_placeholder.write("Querying...")

    if document_file or audio_file or video_file:
        index = load_index_from_storage(
            storage_context=get_storage_context(),
            embed_model=embed_model
        )
        
        query_engine = index.as_query_engine()
        result = query_engine.query(question)
        response_text = result.response

        st.write(response_text)
        querying_placeholder.empty()
        st.empty()

        with st.expander("Details"):
            # Sources
            formatted_sources = result.get_formatted_sources()
            st.write(formatted_sources)

            # Extra info
            st.write(result)

    elif agent:
        result = agent.chat(question)
        st.write(result)
        querying_placeholder.empty()

    elif base64_image:
        chat_completion = vision_model(question, image_content)
        response_text = chat_completion.choices[0].message.content
        st.write(response_text)
        querying_placeholder.empty()

    elif website_link:
        result = smart_scraper_graph.run()
        st.write(result)
        querying_placeholder.empty()

        with st.expander("Execution info"):
            graph_exec_info = smart_scraper_graph.get_execution_info()
            st.write(prettify_exec_info(graph_exec_info))

# Settings
st.subheader("Settings")

delete_button = st.button("Delete Vector Store")
if delete_button:
    shutil.rmtree(VECTOR_STORE_PATH)
    st.success("Deleted Vector Store successfully.")
