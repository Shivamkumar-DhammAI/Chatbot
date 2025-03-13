import os
import faiss
import numpy as np
import re
import nltk
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import gradio as gr
import random
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get all available API keys
def get_api_keys():
    keys = []
    # Look for multiple API keys in format GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, etc.
    i = 1
    while True:
        key = os.getenv(f"GOOGLE_API_KEY_{i}")
        if key:
            keys.append(key)
            i += 1
        else:
            # Also check for the standard GOOGLE_API_KEY
            std_key = os.getenv("GOOGLE_API_KEY")
            if std_key and std_key not in keys:
                keys.append(std_key)
            break
    
    if not keys:
        print("WARNING: No Google API Keys found in environment variables!")
    else:
        print(f"Found {len(keys)} API key(s)")
    
    return keys

# Global variables for API keys
API_KEYS = get_api_keys()
CURRENT_KEY_INDEX = 0

# Function to get the next available API key
def get_next_api_key():
    global CURRENT_KEY_INDEX
    if not API_KEYS:
        return None
    
    CURRENT_KEY_INDEX = (CURRENT_KEY_INDEX + 1) % len(API_KEYS)
    return API_KEYS[CURRENT_KEY_INDEX]

# Download necessary NLTK data
nltk.download("punkt", quiet=True)

# Load transcript file
TRANSCRIPT_FILE = "transcript.txt"

# Read and clean transcript
def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\n", " ")
    return text.strip()

with open(TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
    transcript_text = f.read()

cleaned_text = clean_text(transcript_text)

# Tokenize into sentences
sentences = sent_tokenize(cleaned_text)

# Split into chunks
chunk_size = 500
chunks = []
current_chunk = ""

for sentence in sentences:
    if len(current_chunk) + len(sentence) < chunk_size:
        current_chunk += " " + sentence
    else:
        chunks.append(current_chunk.strip())
        current_chunk = sentence

if current_chunk:
    chunks.append(current_chunk.strip())

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode chunks
chunk_embeddings = np.array([embedding_model.encode(chunk) for chunk in chunks])
chunk_map = {i: chunks[i] for i in range(len(chunks))}

# Save to FAISS index
dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(chunk_embeddings)

# Function to search transcript using FAISS
def search_transcript(query):
    query_embedding = embedding_model.encode([query])
    k = 3  # Number of chunks to retrieve
    distances, indices = index.search(query_embedding, k)
    
    relevant_chunks = [chunk_map[idx] for idx in indices[0] if idx >= 0 and idx < len(chunks)]
    if not relevant_chunks:
        return "No relevant text found."
    return " ".join(relevant_chunks)

# Function to generate AI response with API key rotation on quota error
def generate_response(query):
    global CURRENT_KEY_INDEX
    
    if not API_KEYS:
        return "Error: No API keys available. Please add API keys to your environment variables."
    
    # Try up to the number of available keys
    for _ in range(len(API_KEYS)):
        try:
            # Configure with current API key
            current_key = API_KEYS[CURRENT_KEY_INDEX]
            genai.configure(api_key=current_key)
            model = genai.GenerativeModel("gemini-1.5-pro-latest")
            
            relevant_text = search_transcript(query)
            prompt = f"""
            You are an AI tutor. Answer the following question based on the given lecture transcript:

            Lecture Context: {relevant_text}

            Question: {query}
            """
            response = model.generate_content(prompt)
            return response.text
        
        except Exception as e:
            error_str = str(e).lower()
            # Check if it's a quota error
            if "quota" in error_str or "rate limit" in error_str or "exceeded" in error_str:
                print(f"API key {CURRENT_KEY_INDEX+1} quota exceeded, trying next key...")
                # Get the next key
                CURRENT_KEY_INDEX = (CURRENT_KEY_INDEX + 1) % len(API_KEYS)
            else:
                # For other errors, return the error message
                return f"Error: {str(e)}"
    
    return "All API keys have reached their quota limits. Please try again later."

# Simple interface function
def chatbot(query):
    if not query or query.strip() == "":
        return "Please ask a question about the lecture."
    return generate_response(query)

# Create a simple interface
with gr.Blocks(title="Dhamm AI Chatbot") as demo:
    gr.Markdown("# Dhamm AI Chatbot")
    gr.Markdown("Ask questions about the lecture content")
    
    with gr.Row():
        query_input = gr.Textbox(
            label="Your Question",
            placeholder="Ask anything about the lecture...",
            lines=2
        )
    
    with gr.Row():
        submit_btn = gr.Button("Ask")
        clear_btn = gr.Button("Clear")
    
    with gr.Row():
        output = gr.Textbox(label="Answer", lines=10)
    
    # Add example questions
    gr.Examples(
        examples=["What is this lecture about?", "Can you summarize the key points?"],
        inputs=query_input
    )
    
    # Set up interactions
    submit_btn.click(fn=chatbot, inputs=query_input, outputs=output)
    clear_btn.click(fn=lambda: "", inputs=None, outputs=query_input)

if __name__ == "__main__":
    demo.launch(share=False)