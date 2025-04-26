from dotenv import load_dotenv
from langchain_cohere import CohereEmbeddings, ChatCohere
import os
from langchain import hub
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
load_dotenv()

if __name__ == "__main__":
    print("Retriving.....")
    
    
    embeddings = CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        user_agent="LangChain"
    )
    
    llm = ChatCohere(
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )
    
    query = "What is pinecone?"
    chain = PromptTemplate.from_template(template=query) | llm
    # result = chain.invoke(input={})
    # print(result)
    
    vector_store = PineconeVectorStore(index_name=os.environ["INDEX_NAME"],embedding=embeddings)
    
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=retrieval_qa_chat_prompt, )
    retrieval_chain = create_retrieval_chain(vector_store.as_retriever(),combine_docs_chain)
    result = retrieval_chain.invoke(input={"input": query})
    
    print(result)