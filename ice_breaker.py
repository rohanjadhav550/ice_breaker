from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os

information = """
    Narendra Damodardas Modi[a] (born 17 September 1950) is an Indian politician who has served as the prime minister of India since 2014. Modi was the chief minister of Gujarat from 2001 to 2014 and is the member of parliament (MP) for Varanasi. He is a member of the Bharatiya Janata Party (BJP) and of the Rashtriya Swayamsevak Sangh (RSS), a right-wing Hindu nationalist paramilitary volunteer organisation. He is the longest-serving prime minister outside the Indian National Congress.

    Modi was born and raised in Vadnagar, Bombay State (present-day Gujarat), where he completed his secondary education. He was introduced to the RSS at the age of eight. Modi became a full-time worker for the RSS in Gujarat in 1971. The RSS assigned him to the BJP in 1985 and he rose through the party hierarchy, becoming general secretary in 1998.[b] In 2001, Modi was appointed chief minister of Gujarat and elected to the legislative assembly soon after. His administration is considered complicit in the 2002 Gujarat riots,[c] and has been criticised for its management of the crisis. According to official records, a little over 1,000 people were killed, three-quarters of whom were Muslim; independent sources estimated 2,000 deaths, mostly Muslim.[12] A Special Investigation Team appointed by the Supreme Court of India in 2012 found no evidence to initiate prosecution proceedings against him.[d] While his policies as chief minister were credited for encouraging economic growth, his administration was criticised for failing to significantly improve health, poverty and education indices in the state.[e]
"""

if __name__ == "__main__":
    load_dotenv()
    print("Welcome to LangChain!")
    summary_template = """
        given the information {information} about the person from, I want you to create:
        1. A short summery
        2. two interesting factors about them
    """

    summery_prompt_template = PromptTemplate(
        input_variables=["infromation"], template=summary_template
    )

    # llm = ChatOpenAI(
    #     temperature=0, model="gpt-3.5-turbo", api_key=os.environ["OPENAI_API_KEY"]
    # )
    llm = ChatOllama(temperature=0, model="llama3")

    chain = summery_prompt_template | llm | StrOutputParser()

    res = chain.invoke(input={"information": information})

    print(res)
