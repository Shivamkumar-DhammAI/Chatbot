from flask import Flask, request, jsonify
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import re
import nltk
from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Get API keys
def get_api_keys():
    keys = []
    i = 1
    while True:
        key = os.getenv(f"GOOGLE_API_KEY_{i}")
        if key:
            keys.append(key)
            i += 1
        else:
            std_key = os.getenv("GOOGLE_API_KEY")
            if std_key and std_key not in keys:
                keys.append(std_key)
            break
    return keys

API_KEYS = get_api_keys()
CURRENT_KEY_INDEX = 0

def get_next_api_key():
    global CURRENT_KEY_INDEX
    if not API_KEYS:
        return None
    CURRENT_KEY_INDEX = (CURRENT_KEY_INDEX + 1) % len(API_KEYS)
    return API_KEYS[CURRENT_KEY_INDEX]

nltk.download("punkt", quiet=True)

TRANSCRIPT_FILE = "transcript.txt"

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\n", " ")
    return text.strip()

with open(TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
    transcript_text = f.read()

cleaned_text = clean_text(transcript_text)
sentences = sent_tokenize(cleaned_text)

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

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embeddings = np.array([embedding_model.encode(chunk) for chunk in chunks])
chunk_map = {i: chunks[i] for i in range(len(chunks))}

dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(chunk_embeddings)

def search_transcript(query):
    query_embedding = embedding_model.encode([query])
    k = 3
    distances, indices = index.search(query_embedding, k)
    relevant_chunks = [chunk_map[idx] for idx in indices[0] if idx >= 0 and idx < len(chunks)]
    return " ".join(relevant_chunks) if relevant_chunks else "No relevant text found."

def generate_response(query):
    global CURRENT_KEY_INDEX
    if not API_KEYS:
        return "Error: No API keys available."
    
    for _ in range(len(API_KEYS)):
        try:
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
            if "quota" in error_str or "rate limit" in error_str or "exceeded" in error_str:
                CURRENT_KEY_INDEX = (CURRENT_KEY_INDEX + 1) % len(API_KEYS)
            else:
                return f"Error: {str(e)}"
    return "All API keys have reached their quota limits."

@app.route("/", methods=["GET"])
def home():
    return "Welcome to the Dhamm AI Chatbot API! Use /ask?query=your_question to get answers."

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()  # Get JSON payload from the request
    if not data or "query" not in data:
        return jsonify({"error": "Please provide a query parameter in JSON format."}), 400
    
    query = data["query"]
    response = generate_response(query)
    return jsonify({"answer": response})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4000, debug=True)
