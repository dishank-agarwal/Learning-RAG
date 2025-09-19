from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(web_paths=["https://www.educosys.com/course/genai"])

docs = loader.load()
# print(docs)

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
splits = text_splitter.split_documents(docs)

# print(splits[0])
# print(splits[1])
# print(splits[2])

print(len(splits))

from sentence_transformers import SentenceTransformer
import chromadb

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# splits = text_splitter.split_documents(docs)  # already done in your code

# ---------------------------
# 1. Extract text from split docs
# ---------------------------
texts = [doc.page_content for doc in splits]
# print(texts)

# ---------------------------
# 2. Generate embeddings
# ---------------------------
embeddings = model.encode(texts, convert_to_numpy=True)
# print(embeddings) 

# ---------------------------
# 3. Save into ChromaDB
# ---------------------------
client = chromadb.PersistentClient(path="./chroma_db") # persist file inside -> permenant save (RAM independent)
collection = client.get_or_create_collection(name="docs_embeddings")

collection.add(
    ids=[f"chunk_{i}" for i in range(len(texts))],
    documents=texts,
    embeddings=embeddings.tolist()
)

print(f"✅ Saved {len(texts)} chunks into ChromaDB")

query="How can i learn GEN AI?"

query_embedding = model.encode([query], convert_to_numpy=True)
results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=5
    )

if results["documents"]:
        print("\n🔎 Retrieved Chunks:")
        for i, doc in enumerate(results["documents"][0]):
            print(f"{i+1}. {doc}")
        # For simplicity, return the top chunk as the "answer"
            print(results["documents"][0][0])

# print(results)
print()
print()
print()

prompt = f"Please rephrase the following text to be a clear, concise answer:\n\n{results["documents"]}"

from google import genai

client = genai.Client(api_key="apikey")
response = client.models.generate_content(
        model="gemini-2.0-flash", contents=prompt
    )
if response:
    print(response.text)
