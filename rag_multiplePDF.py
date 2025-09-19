from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
import os
from google import genai
from google.genai import types
from PyPDF2 import PdfReader

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

client_chroma = chromadb.PersistentClient(path="./chroma_db")
collection = client_chroma.get_or_create_collection(name="docs_embeddings")

# addding the gemini client for response generation
gemini_client = genai.Client(api_key="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

for path in os.listdir("./Resumes"):
    reader = PdfReader(os.path.join("./Resumes", path))
    raw_text = ""
    for page in reader.pages:
        raw_text += page.extract_text() + "\n"

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 100,
        chunk_overlap = 50
    )
    chunks = splitter.split_text(raw_text)

    embeddings = embed_model.encode(chunks, convert_to_numpy=True).tolist() # added the emded model after this
    
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        collection.add(
            ids=[f"{path}_{i}"], # for unique IDs
            documents=[chunk],
            embeddings=[emb]
        )
    print(f"✅ Loaded {len(chunks)} chunks into ChromaDB")
    print("Printing the Chunks to check","\n"*2)
    print(chunks,"\n"*4)

chat_history = [] # added after the history

def response_generation(query):
    # added for the query encode
    query_emb = embed_model.encode([query], convert_to_numpy=True)

    # retrieve my chunks for the context
    results = collection.query(
        query_embeddings=query_emb.tolist(),
        n_results=5
    )
    retrieved_chunks = results["documents"][0] if results["documents"] else []

    print("Printing Retireved Chuncks to check","\n"*1)
    print(retrieved_chunks)

    recent = chat_history[-5:]
    history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent])

    context = "\n".join(retrieved_chunks)
    user_prompt = f"""
    You are a helpful assistant.

    Chat history (for context):
    {history}

    User question:
    {query}

    Relevant context from documents:
    {context}

    Task: Provide a clear, concise, user-friendly answer. Rephrase in your own words, don't copy chunks verbatim.
    """
    # generating the response
    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[types.Content(role="user", parts=[types.Part(text=user_prompt)])]
    )

    candidate = response.candidates[0]
    rephrased = "".join([p.text for p in candidate.content.parts if p.text])
    
    return rephrased.strip()


query = "Who is Dishank?"
answer = response_generation(query)

print(f"Answer: {query}\n{answer}")

print("\n"*5)

query = "Tell me about his Projects?"
answer = response_generation(query)

print(f"Answer: {query}\n{answer}")

print("\n"*5)

query = "Who is Dimple?"
answer = response_generation(query)

print(f"Answer: {query}\n{answer}")

print("\n"*5)

query = "Summarize her Resume in aprroximately 60 words?"
answer = response_generation(query)

print(f"Answer: {query}\n{answer}")