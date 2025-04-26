from langchain_community.document_loaders import FireCrawlLoader
from langchain_pinecone import PineconeVectorStore
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()

def ingest_docs() -> None:
    embedding = CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        user_agent="LangChain",
    )
    langchian_docs_urls = [
        "https://langchain-ai.github.io/langgraph/tutorials/introduction/",
    ]

    for url in langchian_docs_urls:
        print(f"Processing {url}")
        loader = FireCrawlLoader(
            url=url,
            mode="crawl",
            params={
                'limit': 1
                }
        )

        docs = loader.load()
        
        text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
        texts = text_splitter.split_documents(docs)
        
        
        print(type(docs))
        
        print(f"Loaded {len(texts)} documents from {url}")
        print("Loading docs into Pinecone...")
        PineconeVectorStore.from_documents(
            texts,
            embedding=embedding,
            index_name=os.getenv("PINECONE_INDEX_NAME"),
        )


if __name__ == "__main__":
    ingest_docs()
