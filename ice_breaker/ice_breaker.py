from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from langchain_community.llms import Cohere
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

# os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")
information = """
Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman known for his key roles in Tesla, SpaceX, PayPal, OpenAI and Twitter (which he rebranded as X). Since 2025, he has been a senior advisor to United States president Donald Trump and the de facto head of the Department of Government Efficiency (DOGE). Musk is the wealthiest person in the world; as of March 2025, Forbes estimates his net worth to be $320 billion USD.

Born to a wealthy family in Pretoria, South Africa, Musk emigrated in 1989 to Canada, whose citizenship he had inherited through his mother. He graduated from the University of Pennsylvania in the U.S. before moving to California to pursue business ventures. In 1995, Musk co-founded the software company Zip2. Following its sale in 1999, he co-founded X.com, an online payment company that later merged to form PayPal, which was acquired by eBay in 2002 for $1.5 billion. That year, Musk also became a U.S. citizen.

In 2002, Musk founded SpaceX and became its CEO and chief engineer. The company has since led innovations in reusable rockets and commercial spaceflight. In 2004, Musk joined Tesla, Inc. as an early investor, and became its CEO and product architect in 2008; it has become a market leader in electric vehicles. In 2015, he co-founded OpenAI to advance artificial intelligence research, but left its board in 2018. In 2016, Musk co-founded Neuralink, a company focused on brain–computer interfaces, and in 2017 launched the Boring Company, which aims to develop tunnel transportation. Musk was named Time magazine's Person of the Year in 2021. In 2022, he acquired Twitter, implementing significant changes and rebranding it as X in 2023. In January 2025, he was appointed head of Trump's newly created DOGE.

Musk's political activities and views have made him a polarizing figure. He has been criticized for making unscientific and misleading statements, including COVID-19 misinformation and promoting conspiracy theories. His acquisition of Twitter (now X) was controversial due to a subsequent increase in hate speech and the spread of misinformation on the service. He has engaged in political activities in several countries, including as a vocal and financial supporter of Trump. Musk was the largest donor in the 2024 U.S. presidential election and is a supporter of global far-right figures, causes, and political parties."""

if __name__ == "__main__":
    print("Hello, World!")
    load_dotenv()
    summary_template = """
    given the information {information} about a person from i want you to create:
    1. A Short Summary
    2. two interesting facts about them"""

    summary_propmt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )
    
    cohere_api_key = os.getenv("COHERE_API_KEY")

    # llm = Cohere(
    #     cohere_api_key=cohere_api_key, 
    #     temperature=0.7,  
    #     max_tokens=500, 
    # )
    
    llm = ChatOllama(
        model="llama3",
    )

    chain = summary_propmt_template | llm | StrOutputParser()

    res = chain.invoke(input={"information": information})

    print(res)
