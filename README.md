# ANLP - Homework

## How to Run
Run the chatbot with:

1. Clone this repository
2. creat python env and install Ollama
3. Pull the required models:
 - ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
 - ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
3. python demo.py
4. Ask any cat-related question!
5. Type exit to quit the program.

## How it works
Reads a list of cat facts from cat-facts.txt.
Converts each fact into an embedding using a HuggingFace model.
Retrieves the top-N most relevant facts in response to a user's question using cosine similarity.
Generates a natural language answer using a small instruction-tuned language model.

## Limitations
knowledge not included, cant be answered

## improvements
use more sufficient vector search like faiss
increase dataset