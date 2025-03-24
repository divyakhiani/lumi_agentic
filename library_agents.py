# region IMPORT STANDARD LIBRARIES
import os
import re
import json
import sqlite3
import openai
import pandas as pd
import yaml
from dotenv import load_dotenv
from pathlib import Path
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
import library_tools as tools
# endregion

# region IMPORT CONFIG LIBRARIES
from config_generic import GenericConfig as Config
from config_logger import logger
# endregion 

# region LOAD CONFIG, API
config = Config()

env_path = Path("credentials/.env")
load_dotenv(dotenv_path=env_path)
openai.api_key = os.getenv("OPENAI_API_KEY")

prompts = libread.read_prompts(file_path=config.prompts)
# endregion

# region load vectorstore and llm 
embedding_model = HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=r"C:\Users\Lenovo\Xtage Technologies Private Limited\Pawas Misra - Divya\Lumi\Codebase - git\lumi_ai\database")
vector_store = Chroma(client=client, collection_name="nilkamal_row_embeddings_ext",embedding_function=embedding_model)
retriever = vector_store.as_retriever()
llm = ChatOpenAI(model_name="gpt-4", temperature=0)
# endregion 

# region Persistence layer
# conn = sqlite3.connect("chat_history/history.db")
# cursor = conn.cursor()

# # Create table if not exists
# cursor.execute("""
# CREATE TABLE IF NOT EXISTS chat_history (
#     session_id TEXT,
#     query TEXT,
#     response TEXT,
#     record_create_time DATETIME DEFAULT CURRENT_TIMESTAMP
# )
# """)
# conn.commit()
# endregion 

class State(BaseModel):
    query: str
    root_product: str           # product_id
    session_id: str
    reformatted_query: str = ""
    intent: str = ""      # action_required
    context: str = ""
    history: list = []  # Last 5 interactions
    summary: str = ""  # Summary of older chat
    response: str = ""
    retry_count: int = 0 
    
# region Query Manager
class QueryManagerAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = prompts['query_manager']['intent_classifier']['prompt']

    @libdec.log_function
    def query_process(self, state):
        user_query = state.query
        product_id = state.root_product
        
        intent = self.llm.invoke(self.prompt.format(query=user_query, tools=prompts['tools']['available_tools'])).content.strip()
        
        prompt = prompts['query_manager']['reformat_query']['prompt'].format(product_id=product_id, user_query=user_query)

        reformatted_query = self.llm.invoke(prompt).content.strip()
        
        logger.info(f"REFORMATED QUERY ==> {reformatted_query}")
        logger.info(f"INTENT ==> {intent}")
        return {"intent": intent, "reformatted_query": reformatted_query}
# endregion

# region History Manager 
class HistoryManagerAgent:
    def __init__(self, llm, db_path="chat_history/history.db"):
        self.db_path = db_path
        self._initialize_db()
        self.llm = llm

    def _initialize_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                query TEXT,
                response TEXT,
                record_create_time DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    @libdec.log_function
    def update_history(self, state):
        session_id = state.session_id
        query = state.query
        response = state.response
        
        # logger.debug(type(session_id))
        # logger.debug(type(query))
        # logger.debug(type(response))
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO chat_history (session_id, query, response) VALUES (?, ?, ?)", (session_id, query, response))
        conn.commit()
        conn.close()

        logger.info(f"History updated for session {session_id}")
        return {}
    @libdec.log_function
    def history_process(self, state):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT query, response FROM chat_history WHERE session_id = ? ORDER BY record_create_time DESC LIMIT 5", (state.session_id,))
        history = cursor.fetchall()
        
        cursor.execute("SELECT query, response FROM chat_history WHERE session_id = ? ORDER BY record_create_time DESC LIMIT 20 OFFSET 5", (state.session_id,))
        older_chats = cursor.fetchall()
        conn.close()

        if older_chats and self.llm:
            chat_text = "\n".join([f"Q: {q}\nA: {r}" for q, r in older_chats])
            summary_prompt = f"Summarize the following chat history:\n{chat_text}"
            summary = self.llm.predict(summary_prompt).strip()
        else:
            summary = ""
            
        logger.info(f"HISTORY ==> {history}")
        logger.info(f"SUMMARY ==> {summary}")

        return {"history": history, "summary": summary}
# endregion

# region Context Manager
class ContextManagerAgent:
    def __init__(self, retriever):
        self.retriever = retriever

    @libdec.log_function
    def context_process(self, state):
        intent = state.intent
        reformatted_query = state.reformatted_query
        product_id = state.root_product
        
        if intent == "Product Info":
            retrieved_data = self.retriever.invoke(f"product information for {product_id}")
        else:
            retrieved_data = self.retriever.invoke(reformatted_query)
            
        metadata_list = [doc.metadata for doc in retrieved_data] 
        
        logger.info(f"CONTEXT ==> {metadata_list}")
        
        return {"context": metadata_list}

# endregion

# region Action Manager
class ActionManagerAgent:
    def __init__(self):
        self.tools = {
            "Product Info": tools.fetch_product_info,
            "Alternate Product": tools.alternate_product_recommendations
        }
        self.fallback_tool = tools.general_query_handler 

    @libdec.log_function
    def action_process(self, state):
        intent = state.intent

        if intent in self.tools:
            tool_function = self.tools.get(intent, self.fallback_tool)
            response = tool_function(
                retrieved_context=state.context, 
                history=state.history, 
                summary=state.summary,
                user_query=state.query,
                root_product=state.root_product
                )
        else:
            logger.debug(f"UNKNOWN INTENT {intent}")
            response = "Sorry, I couldn't understand your request."

        return {"response": response}

# endregion

# region Validate Response
class ValidateResponseAgent:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = prompts['validate_response']['validator']['prompt']

    @libdec.log_function
    def validate_response_process(self, state):
        user_query = state.query
        generated_response = state.response
        
        relevance_check = self.llm.invoke(self.prompt.format(query=user_query, response=generated_response)).content.strip().lower()
        
        if "yes" in relevance_check:
            return {}
        
        if "no" in relevance_check:
            state.retry_count += 1
            return {}

        
# endregion 

# region Initialize agents
query_manager_agent = QueryManagerAgent(llm)
context_manager_agent = ContextManagerAgent(retriever)
action_manager_agent = ActionManagerAgent()   # tools decision 
history_manager_agent = HistoryManagerAgent(llm) 
validate_response_agent = ValidateResponseAgent(llm)
# endregion 

# region Graph 
graph = StateGraph(State)

graph.add_node("query_manager", query_manager_agent.query_process)
graph.add_node("context_manager", context_manager_agent.context_process)  # 
graph.add_node("history_manager", history_manager_agent.history_process)
graph.add_node("action_manager", action_manager_agent.action_process)
graph.add_node("history_update", history_manager_agent.update_history)
graph.add_node("validate_response", validate_response_agent.validate_response_process)
graph.add_node("end", lambda state: state)  # This node just returns state as is

# Define Transitions
graph.set_entry_point("query_manager")
graph.add_edge("query_manager", "context_manager")
# graph.add_edge("context_manager", "history_update")
# graph.add_edge("history_update", "history_manager")
graph.add_edge("context_manager", "history_manager")
graph.add_edge("history_manager", "action_manager")
graph.add_edge("action_manager", "history_update")
graph.add_edge("action_manager", "validate_response")

def retry_decision(state):
    return "query_manager" if state.retry_count == 1 else "end"

graph.add_conditional_edges(
    "validate_response",
    retry_decision,
    {
        "query_manager": "query_manager",  # Retry once
        "end": "end"
    }
)

graph = graph.compile()
# endregion 

# Chatbot Interface
def chatbot(user_query, product_id, session_id):
    state = State(
        query=user_query, 
        root_product=product_id, 
        session_id=session_id
        )
    final_state = graph.invoke(state)
    return final_state.get("response", "No response generated.")


# Example Run
if __name__ == "__main__":
    # user_query = "How old is nilkamal ?"
    # user_query = "What is the warranty period?"
    user_query = "Give me an automatic recliner ?"
    session_id = "trial_1234" 
    # product_id = "Nilkamal Platinum Plastic Arm Chair (Season Rust Brown)"
    product_id = "Nilkamal Sierra 1 Seater Manual Recliner Sofa (Brown)"
    response = chatbot(
        user_query=user_query, 
        product_id=product_id, 
        session_id=session_id
        )
    print("Bot:", response)