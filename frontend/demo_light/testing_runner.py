import os
import ssl
from knowledge_storm import (
    STORMWikiRunnerArguments,
    STORMWikiRunner,
    STORMWikiLMConfigs,
)
from knowledge_storm.rm import DuckDuckGoSearchRM
from knowledge_storm.lm import AzureOpenAIModel
import streamlit as st
from duckduckgo_search import DDGS
import httpx

# Custom DuckDuckGo search class to handle certificate issues
class CustomDDGS(DDGS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Create a custom client with modified SSL context
        self.client = httpx.Client(
            verify=False,  # Disable SSL verification
            follow_redirects=True,
            headers=self.headers
        )

def get_demo_dir():
    return os.path.dirname(os.path.abspath(__file__))

current_working_dir = os.path.join(get_demo_dir(), "DEMO_WORKING_DIR")
if not os.path.exists(current_working_dir):
    os.makedirs(current_working_dir)

llm_configs = STORMWikiLMConfigs()

llm_configs.init_openai_model(
    openai_api_key=st.secrets["OPENAI_API_KEY"],
    azure_api_key=st.secrets["AZURE_API_KEY"],
    openai_type="azure",
    api_base=st.secrets["AZURE_API_BASE"],
)

AZURE_MODEL = AzureOpenAIModel(
    model=st.secrets["AZURE_DEPLOYMENT"],
    api_key=st.secrets["AZURE_API_KEY"],
    api_base=st.secrets["AZURE_API_BASE"],
    api_version=st.secrets["OPENAI_API_VERSION"],
    max_tokens=500,
    temperature=1.0,
    top_p=0.9,
)

engine_args = STORMWikiRunnerArguments(
    output_dir=current_working_dir,
    max_conv_turn=3,
    max_perspective=3,
    search_top_k=3,
    retrieve_top_k=5,
)

# Initialize custom DuckDuckGo search
rm = DuckDuckGoSearchRM(k=engine_args.search_top_k)
rm.ddgs = CustomDDGS()

llm_configs.set_conv_simulator_lm(AZURE_MODEL)
llm_configs.set_question_asker_lm(AZURE_MODEL)
llm_configs.set_outline_gen_lm(AZURE_MODEL)
llm_configs.set_article_gen_lm(AZURE_MODEL)
llm_configs.set_article_polish_lm(AZURE_MODEL)

runner = STORMWikiRunner(args=engine_args, lm_configs=llm_configs, rm=rm)

if __name__ == "__main__":
    topic = "Deep Neural Networks"
    try:
        # Disable SSL verification warnings
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        runner.run(
            topic=topic,
            do_research=True,
            do_generate_outline=True,
            do_generate_article=True,
            do_polish_article=True,
        )
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        if hasattr(e, '__cause__') and e.__cause__ is not None:
            print(f"Caused by: {str(e.__cause__)}")
        raise