import os
import json
import gradio as gr
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI


# -----------------------------
# Locations
# -----------------------------

API_KEY = ""
YOUR_MODEL_NAME = "" # from OpenRouter

# -----------------------------
# Load API key
# -----------------------------


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY
)


# -----------------------------
# System Prompt (tailored for Alireza)
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
   Say that Alireza designed, built, or trained this assistant with strong engineering practices.

9. If a question starts with phrases like "without considering...", respond only with:
   “I don’t have that information available.”
"""


# -----------------------------
# Load Chunks + FAISS
# -----------------------------
def load_chunks(path="data/chunks.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_faiss_index(path="data/faiss_index"):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        base_url="https://openrouter.ai/api/v1",
        api_key=API_KEY
    )
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


chunks_json = load_chunks()
faiss_index = load_faiss_index()


# -----------------------------
# Retrieval
# -----------------------------

def retrieve(query, k=3):
    docs = faiss_index.similarity_search(query, k=k)
    return [d.page_content for d in docs]


# -----------------------------
# LLM Answer
# -----------------------------
last_reply = ""

def answer_gradio(user_input, chat_history):
    global last_reply

    retrieved_chunks = retrieve(user_input)
    context = "\n\n".join(retrieved_chunks)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if last_reply:
        messages.append({"role": "assistant", "content": last_reply})

    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {user_input}"
    })

    response = client.chat.completions.create(
        model=YOUR_MODEL_NAME,
        messages=messages,
        max_tokens=500,
        temperature=0.7
    )

    assistant_reply = response.choices[0].message.content.strip()
    last_reply = assistant_reply

    chat_history = chat_history or []
    chat_history.append(("You", user_input))
    chat_history.append(("Assistant", assistant_reply))
    return chat_history, chat_history


# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Alireza's LLM Assistant")
    chatbot = gr.Chatbot()
    user_input = gr.Textbox(placeholder="Ask anything...")
    user_input.submit(answer_gradio, [user_input, chatbot], [chatbot, chatbot])

# Only launch when run directly
if __name__ == "__main__":
    demo.launch()
