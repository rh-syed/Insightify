from dotenv import load_dotenv
from langchain.chains import LLMChain

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from third_parties.linkedin_scraper import scrape_linkedin_profile

if __name__ == "__main__":
    load_dotenv()

    print("Insightify Initiated")

    summary_template = """
        given the LinkedIn information {information} about a person I want you to create:
        1. A short summary
        2. two interesting facts about them
        """
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    linkedin_profile_url = linkedin_lookup_agent(
        "Yoshie Matsuura - Toronto, Canada - TDSB - Teacher"
    )

    linked_in_data = scrape_linkedin_profile(linkedin_profile_url)

    # print(linkedin_profile_url)

    # print(linked_in_data)

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)
    res = chain.invoke(input={"information": linked_in_data})

    print(res)
