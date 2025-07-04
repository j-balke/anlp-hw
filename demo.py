import ollama

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

def get_dataset():
    with open('cat-facts.txt', 'r', encoding="utf-8") as file:
        dataset = [line.strip() for line in file]
    return dataset

def create_vector_db(dataset):
    VECTOR_DB = []

    def add_chunk_to_database(chunk):
        embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
        VECTOR_DB.append((chunk, embedding))
    
    for i, chunk in enumerate(dataset):
        add_chunk_to_database(chunk)
    
    return VECTOR_DB

def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(y ** 2 for y in b) ** 0.5
    return dot_product / (norm_a * norm_b)

def retrieve(vector_db, query, top_n=3):
  query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
  similarities = []
  for chunk, embedding in vector_db:
    similarity = cosine_similarity(query_embedding, embedding)
    similarities.append((chunk, similarity))
  # sort by similarity in descending order, because higher similarity means more relevant chunks
  similarities.sort(key=lambda x: x[1], reverse=True)
  # finally, return the top N most relevant chunks
  return similarities[:top_n]

def get_instruction(retrieved_knowledge):
    context_chunks = '\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])
    instruction_prompt = f'''You are a helpful chatbot.
        Use only the following pieces of context to answer the question. Don't make up any new information:
        {context_chunks}'''
    return instruction_prompt

def generate_answer(user_input, instruction_prompt):
    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {'role': 'system', 'content': instruction_prompt},
            {'role': 'user', 'content': user_input},
        ],
        stream=True,)
    
    return stream

def print_resonse(stream, top_n):
    print(f"Chatbot response for {top_n} retrievals:")
    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)
    print("")
    print(50 * "-")
    return

def main():
    dataset = get_dataset()
    vector_db = create_vector_db(dataset)

    num_retrievals = [1,3,5]

    print("Hello! I am a chatbot that can answer questions about cats")
    print("Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("Ask me a question?")
        if user_input.lower() == "exit":
            print("chatbot is closed.")
            break
        
        for top_n in num_retrievals:

            retrieved_knowledge = retrieve(vector_db, user_input, top_n)

            instruction_prompt = get_instruction(retrieved_knowledge)
            stream = generate_answer(user_input, instruction_prompt)
            print_resonse(stream, top_n)

if __name__ == "__main__":
    main()
