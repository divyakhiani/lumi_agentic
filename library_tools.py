# region IMPORT STANDARD LIBRARIES
import os
import re
import json
import sqlite3
import openai
import pandas as pd
import yaml
# endregion

# region IMPORT THIRD-PARTY LIBRARIES
import langchain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langgraph.graph import StateGraph
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os 
import chromadb
from pydantic import BaseModel
# endregion 

# region IMPORT CUSTOM LIBRARIES
import library_decorators as libdec
import library_read_files as libread
# endregion

# region IMPORT CONFIG LIBRARIES
from config_generic import GenericConfig as Config
from config_logger import logger
# endregion 

# region LOAD CONFIG, API
config = Config()
os.environ["OPENAI_API_KEY"] = config.models['openai']['key_001']
prompts = libread.read_prompts(file_path=config.prompts)
# endregion

# region load vectorstore and llm 
embedding_model = HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=r"C:\Users\Lenovo\Xtage Technologies Private Limited\Pawas Misra - Divya\Lumi\Codebase - git\lumi_ai\database")
vector_store = Chroma(client=client, collection_name="nilkamal_row_embeddings_ext",embedding_function=embedding_model)
retriever = vector_store.as_retriever()
llm = ChatOpenAI(model_name="gpt-4", temperature=0)
# endregion 


def fetch_product_info(**kwargs):
    logger.info("FETCHING PRODUCT INFORMATION")
    product_id = kwargs.get('product_id', None)
    user_query = kwargs.get('user_query', None)
    retrieved_context = kwargs.get('retrieved_context', None)
    
    prompt = f"""User is looking at {product_id}.
                 The user's question is: {user_query}.
                 The retrieved context is: {retrieved_context}.
                 Please answer the user's question.
            """
            
    response = llm.invoke(prompt)
    logger.debug(f"PRODUCT INFO ==> {response}")
    return response.content

def alternate_product_recommendations(**kwargs):
    logger.info("ALTERNATE PRODUCT RECOMMENDATIONS")

def general_query_handler(**kwargs):
    user_query = kwargs.get("user_query", "No query provided")
    
    # Log the unknown intent for debugging
    logger.warning(f"Unhandled query: {user_query}")

    # Option 1: Provide a polite fallback response
    return "I'm not sure how to handle this request. Could you clarify?"

    # Option 2 (Advanced): Use an LLM to generate a response
    # llm_response = llm.invoke(f"Provide a helpful response to: {user_query}")
    # return llm_response.strip()

    
# region TOOL TRIAL
def compare_products():
    logger.info("COMPARING PRODUCTS")
    return {"response": "Nilkamal is the best"}

def find_similar_products():
    logger.info("FINDING SIMILAR PRODUCTS")
    return {"response": "Nilkamal Sierra recliner and Nilkamal Chair are similar"}

def recommend_accessories():
    logger.info("RECOMMENDING ACCESSORIES")
    return {"response" : "Nilkamal Coffee table is a great accessory for Nilkamal Sierra recliner"}

def unclassified(user_query):
    logger.info("UNCLASSIFIED INTELLIGENCE")
    response = llm.invoke(user_query)
    return {"response": response}

# endregion