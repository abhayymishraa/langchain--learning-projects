
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_cohere import CohereEmbeddings, ChatCohere
import os
from langchain import hub
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
if __name__ == "__main__":
    print("Chat with PDF.....")
    pdf_path = "bitcoin_white_paper.pdf"
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents)
    
    embeddings = CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        user_agent="LangChain"
    )
    
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local("faiss_index")
    
    new_vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    llm = ChatCohere(
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )
    
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm= llm, prompt=retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(new_vector_store.as_retriever(),combine_docs_chain=combine_docs_chain)
    
    res = retrieval_chain.invoke(input={"input": "What is Bitcoin White Paper? Why is it used for? What is the use case? "})
    
    print(res)