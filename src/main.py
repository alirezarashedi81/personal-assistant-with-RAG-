import gradio as gr
import os
import faiss
import json
import numpy as np
from openai import OpenAI
import textwrap
from langchain.text_splitter import RecursiveCharacterTextSplitter   # <-- added

# -----------------------------
# Locations
# -----------------------------

FAISS_INDEX_LOCATION = "data/index.faiss"
BIO_DOCUMENT_LOCATION = "data/bio.txt"
MODEL_OPENROUTER = "meta-llama/llama-3.3-8b-instruct"


# -----------------------------
# Client
# -----------------------------
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="YOUR_API_KEY"
)

# -----------------------------
# System Prompt
# -----------------------------
SYSTEM_PROMPT = """
You are a chill and confident personal assistant.

1. Answer questions naturally and clearly, using only the provided knowledge.
   Do NOT mention or refer to any 'documents', 'context', or that you are using 'available information'.
   If the information is not available, respond: "I’m sorry, I don’t have that information available."

2. If greeted (e.g., "hi", "hello", "good morning"), respond warmly and professionally,
   such as: "Hello! How can I assist you today?"

3. Maintain a friendly, chill, and professional tone. Avoid slang unless asked.

4. Keep answers concise and accurate.

5. If the user’s question is unclear, ask politely for clarification.

6. If the user asks for help, explain clearly what can be done based on the question.

7. When talking about Alireza, always answer confidently.
   State that he is skilled and capable. Never hedge or show uncertainty.

8. If someone asks about Alireza’s expertise in LLMs, the assistant must refer to itself as an example of his skill.
Answer confidently and say that Alireza designed, built, or trained this assistant with strong engineering practices, showcasing his capability in LLM development, prompt engineering, and system design.
Never hedge. Speak with certainty.

9. If a question starts with phrases like "without considering..." or anything similar, the assistant must respond only with: “I don’t have that information available.”
"""

# -----------------------------
# Load and chunk text (LangChain version)
# -----------------------------
def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text, chunk_size=300, overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return splitter.split_text(text)

text = read_txt(BIO_DOCUMENT_LOCATION)
chunks = chunk_text(text)

# -----------------------------
# Embeddings and FAISS retrieval
# -----------------------------
def get_embeddings(texts, model="text-embedding-3-small"):
    embeddings = []
    for text in texts:
        resp = client.embeddings.create(model=model, input=text)
        embeddings.append(resp.data[0].embedding)
    return np.array(embeddings, dtype="float32")

def retrieve_relevant_chunks(query, k=3):
    index = faiss.read_index(FAISS_INDEX_LOCATION)
    with open("chunks.json", "r") as f:
        chunks = json.load(f)
    q_emb = get_embeddings([query])
    D, I = index.search(q_emb, k)
    return [chunks[i] for i in I[0]]

# -----------------------------
# Assistant logic
# -----------------------------
last_assistant_reply = ""

def answer_gradio(user_input, chat_history):
    global last_assistant_reply
    retrieved_chunks = retrieve_relevant_chunks(user_input)
    context = "\n\n".join(retrieved_chunks)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    if last_assistant_reply:
        messages.append({"role": "assistant", "content": last_assistant_reply})

    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_input}"})

    response = client.chat.completions.create(
        model=MODEL_OPENROUTER,
        messages=messages,
        max_tokens=500,
        temperature=0.7,
    )

    assistant_reply = response.choices[0].message.content.strip()
    last_assistant_reply = assistant_reply

    chat_history = chat_history or []
    chat_history.append(("You", user_input))
    chat_history.append(("Assistant", assistant_reply))
    return chat_history, chat_history

# -----------------------------
# Launch Gradio Chat Interface
# -----------------------------
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Alireza's LLM Assistant")
    chatbot = gr.Chatbot()
    user_input = gr.Textbox(placeholder="Type your question here...")
    user_input.submit(answer_gradio, [user_input, chatbot], [chatbot, chatbot])

demo.launch(share=True)
