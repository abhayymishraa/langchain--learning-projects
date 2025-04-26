from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from typing import List, Dict, Any
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_cohere import CohereEmbeddings, ChatCohere
import os
from langchain import hub
from langchain_pinecone import PineconeVectorStore


load_dotenv()


def runllm(query: str, chat_history: List[Dict[str, any]] = []):
    embeddings = CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        user_agent="LangChain",
    )

    docsearch = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME"), embedding=embeddings
    )

    chat = ChatCohere(verbose=True, cohere_api_key=os.getenv("COHERE_API_KEY"))

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    stuff_document_chain = create_stuff_documents_chain(
        chat, prompt=retrieval_qa_chat_prompt
    )

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    history_aware_retriever = create_history_aware_retriever(
        llm=chat,
        retriever=docsearch.as_retriever(),
        prompt=rephrase_prompt,
    )

    qa = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=stuff_document_chain,
    )

    result = qa.invoke(input={"input": query, "chat_history": chat_history})

    return result
