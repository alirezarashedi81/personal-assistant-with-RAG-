# Personal Assistant with RAG

A customizable LLM assistant deployed with Gradio, compatible with any OpenRouter model and ready to run on cloud platforms such as Azure. Users can upload their own bio to instantly generate a personalized AI assistant. This version includes a system prompt specifically tailored for Alireza, demonstrating a unique personalized behavior and tone.

---

## Features

- Works with any OpenRouter-supported model.
- Upload a bio file and instantly generate a personalized assistant.
- RAG-based retrieval system using FAISS vector search.
- Gradio UI for easy deployment and interaction.
- Fully compatible with cloud GPU servers and Azure deployments.
- System prompt preconfigured for Alireza (customizable).

---

## How It Works

1. Your bio is processed and split using `RecursiveCharacterTextSplitter`.
2. Text chunks are converted into embeddings using `OpenAIEmbeddings`.
3. FAISS builds a vector index and retrieves relevant information for each query.
4. The assistant uses both retrieved context and your system prompt to answer.
5. Gradio serves a ready-to-use interface through `launch_ui()`.

---

## Installation

```bash

pip install -r requirements.txt

```
Make sure you set your OpenRouter API key as an environment variable:


```Copy code below:
API_KEY "your_key_here"
```
Or inside the script:

```
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="your_api_key"
)

```
