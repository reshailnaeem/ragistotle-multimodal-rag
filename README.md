## RAGistotle - The All-in-One Multimodal RAG

>_"Omnēs hominēs nātūrā scīre RAGs dēsiderant."_
>
>All men by nature desire to know RAGs.
>
>_-RAGistotle, 2024 A.D._
>
>(ChatGPT translated Latin, of course)

Why _"RAGistotle"_? Because, like Aristotle, who mastered many fields, RAGistotle is an all-in-one Retrieval-Augmented Generation (RAG) application that supports diverse input types: text files, audio, video, images, data formats, YouTube links, and website links. It's a versatile tool where each format can be queried in multiple ways. The app uses different library integrations for each task:
- [LlamaIndex](https://www.llamaindex.ai/) as the primary engine, along with [PandasAI](https://pandas-ai.com/) for data formats, and [ScrapeGraphAI](https://scrapegraphai.com/) for the websites
- [Groq](https://groq.com/) (vanilla) and [LangChain Groq](https://python.langchain.com/docs/integrations/chat/groq/) for the LLM models (can also be used with OpenAI models or other LLMs)
- [Qdrant](https://qdrant.tech/) as the local database for vector storage
- [all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) for sentence embedding
- [Pillow](https://python-pillow.org/) for converting images to Base64 encoding for an LLM-friendly format
- [`pytubefix`](https://github.com/JuanBindez/pytubefix) for YouTube links
- [Streamlit](https://streamlit.io/) for the UI

As for the Groq models used, I use [Gemma 2.9B IT](https://huggingface.co/google/gemma-2-9b-it) for general querying, [Whisper Large v3 Turbo](https://huggingface.co/openai/whisper-large-v3-turbo) for audio, and [Llama 3.2 90B Vision](https://huggingface.co/meta-llama/Llama-3.2-90B-Vision) for images.

## Table of Contents
- [Features](#features)
- [Usage](#usage)
- [Limitations](#limitations)
- [License](#license)

## Features
Here’s the story of how RAGistotle came to be. I kept seeing RAGs that could chat with individual file formats - mainly, they’d summarize the content. But I wanted something different. That's when I got the idea for an all-in-one RAG that could accept different file types in a single, unified experience and maybe do more than just summarization. And thus, RAGistotle was born. Yes, there are other apps out there that do similar things, but, of course, they aren’t RAGistotle.

The RAG accepts the following formats:

- **Documents**: Text files, DOC/DOCX, PDFs
- **Audio**: MP3, WAV files
- **Video**: MP4
- **Images**: JPG, BMP, PNG
- **Data Files**: CSV and Excel files
- **Web**: YouTube links and website links

## Usage
Clone the repository:
```bash
git clone https://github.com/reshailnaeem/ragistotle-multimodal-rag.git
cd ragistotle-multimodal-rag
```

Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

Use any Python >= 3.11 version along with the requirements.txt file in the repository. I used 3.12 with the latest versions of all libraries.

```bash
# Installing using repo file
pip install -r requirements.txt

# Install binary distribution to ensure proper compatibility with newer Python versions
pip install --only-binary :all: pandasai

playwright install
```

You'll need API keys from Groq and PandasAI. Use environment variables or the `python-dotenv` library to set them up.

```bash
export GROQ_API_KEY=<YOUR_GROQ_API_KEY>
export PANDASAI_API_KEY=<YOUR_PANDASAI_API_KEY>
```

Run the app using:
```bash
streamlit run app.py
```

You can now access the app locally on:
```bash
http://localhost:8501/
```

## Limitations
RAGistotle is a work in progress; I'm going to keep improving it whenever I get the time. Right now, these are some of the limitations and/or issues currently being refined. Any contributions are more than welcome!

- Ollama support for local LLM support and data privacy
- Deployed version for users to use with their own API keys
- Working on adding support for other LLMs, like OpenAI and Anthropic
- Support for multiple files where libraries allow support
- Cloud support for vector storage
- Sometimes, the app returns parsing errors when scraping. A simple fix is to add "Give the result in JSON format compatible with the output schema" at the end of the query to ensure correct output parsing

## License

This project is licensed under the GNU General Public License v3.0 (GPLv3). See the LICENSE file for details.
