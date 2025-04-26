from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_cohere import CohereEmbeddings
import os
from langchain_pinecone import PineconeVectorStore

# https://python.langchain.com/api_reference/cohere/embeddings/langchain_cohere.embeddings.CohereEmbeddings.html
# pinecone is used to vector store (free tier available)
# notes :- https://handsomely-skunk-c10.notion.site/LLM-1c59427eac9a804099d4e4e367dbacf3?pvs=4
# desiarilization attack

load_dotenv()


if __name__ == "__main__":
    print("Ingestion module")

    loader = TextLoader("medium_blog.txt")
    document = loader.load()
    print("Document loaded")
    print("splitting document into chunks")
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
    )
    texts = text_splitter.split_documents(document)
    print(f"creatred {len(texts)} chunks")
    
    embeddings = CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        user_agent="LangChain"
    )
    
    print("ingesting chunks into vector store")
    
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.getenv("INDEX_NAME"))
    print("finished ingesting chunks into vector store")    