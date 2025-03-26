# region IMPORT STANDARD LIBRARIES
from ast import literal_eval
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
from langchain_groq import ChatGroq
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
df_products = libread.read_data(file = config.file_name_products)

try:
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    response = llm.invoke("Hi")
except Exception as e:
    logger.error(f"Failed to initialize OPEN AI LLM: {e}")
    logger.info("USING CHAT GROQ INSTEAD")
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
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
    user_id: str
    product_info: str = ""
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
        self.intent_thought = ""

    @libdec.log_function
    def identify_intent(self, state):
        user_query = state.query
        product_id = state.root_product
        chat_history = ""
        
        thought_intent = self.llm.invoke(self.prompt.format(
            persona=persona,
            business=business,
            purpose=purpose,
            predefined_intents=predefined_intents,
            instructions=instructions,
            examples=examples,
            user_query=user_query,
            chat_history=chat_history,
            root_product=product_id,
            product_description=product_description)).content.strip()
       
        thought_intent = literal_eval(thought_intent)
        logger.info(f"INTENT ==> {thought_intent}")
        self.intent_thought = thought_intent['thought']
        return {"intent": thought_intent['intent']}
    
    def reformat_query(self, state):
        product_id = state.root_product
        user_query = state.query
        
        prompt = prompts['query_manager']['reformat_query']['prompt'].format(
            thought = self.intent_thought,
            intent = state.intent,
            root_product=product_id, 
            user_query=user_query,
            )
        
        reformatted_query = self.llm.invoke(prompt).content.strip()

        logger.info(f"REFORMATED QUERY ==> {reformatted_query}")
        return  {"reformatted_query": reformatted_query}
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
            summary = self.llm.invoke(summary_prompt).content.strip()
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

    def search_product(
        self,
        query, 
        df_products=df_products
    ):
        """
        Searches the DataFrame for relevant product and appends the required information for building product information.
        Returns the most relevant review details as a string.
        """
        product_columns = df_products.columns.to_list()
        columns_to_remove = ['Title', 'Images', 'URL', 'Similar Products']

        product_columns = [column for column in product_columns if column not in columns_to_remove]
        # print(product_columns)
        # Check if any product name contains keywords from the query
        matching_products = df_products[df_products['Title'].fillna("").astype(str).str.contains(query.strip(), case=False, na=False, regex=False)]
        
        if matching_products.empty:
            return "Sorry, I couldn't find a matching product."

        # Get the most relevant product (first match for simplicity)
        product = matching_products.iloc[0]

        # Prepare response with available details
        response = f"**{product['Title']}**\n"

        for column in product_columns:
            if pd.notna(product[column]):
                response += f"{column}: {product[column]}\n"
                
        return response

    @libdec.log_function
    def get_product_info(self, state):
        product_id = state.root_product
        
        product_info = self.search_product(query=product_id)
        
        # logger.debug(f"Root Product Info ==> {product_info}")
        
        return {"product_info": product_info}

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
        
        # logger.info(f"CONTEXT ==> {metadata_list}")
        
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

# region variables
persona = "intelligent shopping assistant"
business = "online furniture store"
purpose = "to classify the intent of a customer's query"
predefined_intents = """1. **Simple Product Info**  
						2. **Direct Product Comparison**  
						3. **Needs Convincing**  
						4. **Alternate Product Opportunity**  
						5. **Upsell Opportunity**  
						6. **Cross-sell Opportunity**  
						7. **Simple Category Info**  
						8. **Simple Company Info**  
						9. **Unclassified**  
						"""

examples = """#### **Example 1:**  
**Query:** *What is the warranty of the product?*  
**Thought:** This is a straightforward question about product details. The customer is already interested in this product, so there is no need to push an alternative.  
**Intent:** **Simple Product Info**  

#### **Example 2:**  
**(Chatbot previously recommended another product)**  
**Query:** *Why have you recommended this product?*  
**Thought:** Since the customer was already considering a product but we recommended another, they likely want a clear **comparison** to justify the recommendation.  
**Intent:** **Direct Product Comparison**  

#### **Example 3:**  
**Query:** *How is the ergonomics of this sofa?*  
**Thought:** While I can provide product descriptions about ergonomics, real customer reviews reinforcing positive aspects would be more persuasive. Sharing a relevant customer review can help the customer make a decision.  
**Intent:** **Needs Convincing**  

#### **Example 4:**  
**Query:** *Is this available in leather?*  
**Thought:** The customer is interested in a variation of the same product in a different material. If a similar product exists in leather, I should suggest it.  
**Intent:** **Alternate Product Opportunity**  

#### **Example 5:**  
**Query:** *Does this come in a bigger size?*  
**Thought:** The customer is looking for an upgraded version of the same product. If a premium model is available, I should pitch it.  
**Intent:** **Upsell Opportunity**  

#### **Example 6:**  
**Query:** *Do you have a coffee table that matches this sofa?*  
**Thought:** The customer is looking for a complementary product that goes with their main purchase. I should suggest related items.  
**Intent:** **Cross-sell Opportunity**  

#### **Example 7:**  
**Query:** *What are some good sofa brands you offer?*  
**Thought:** The customer is asking about a product category in general rather than a specific product.  
**Intent:** **Simple Category Info**  

#### **Example 8:**  
**Query:** *Where is your company based?*  
**Thought:** The customer is asking about the company rather than a product.  
**Intent:** **Simple Company Info**  

#### **Example 9:**  
**Query:** *I need something different, but I don’t know what exactly.*  
**Thought:** The query is vague and does not fit into any specific category. More clarification is needed.  
**Intent:** **Unclassified**  
"""

instructions = """- If the customer asks a **factual detail** about a product, classify it as **Simple Product Info**.  
- If the customer expresses doubt about the product and needs **validation**, classify it as **Needs Convincing**.  
- If the customer asks **why another product was suggested**, they likely seek **Direct Product Comparison**.  
- If the customer’s query suggests they are **not fully convinced** or **maybe looking for product/s other than the current root product** and may need a **different product**, classify it as **Alternate Product Opportunity**.  
- If the customer is looking for specific features or feature belong to a different category/sub-category which are not available in the root product classify it as **Alternate Product Opportunity**.
- If there is a **higher-tier product** that better meets their needs, classify it as **Upsell Opportunity**.  
- If the conversation suggests they may benefit from an **additional product** (e.g., accessories, related items), classify it as **Cross-sell Opportunity**.  
- If they ask about **product categories** in general, classify it as **Simple Category Info**.  
- If they inquire about **the company, policies, or brand**, classify it as **Simple Company Info**.  
- If the intent is **ambiguous**, classify it as **Unclassified**.  

"""

product_description = """Product Name: Nilkamal Sierra Engineered Wood 1-Seater Recliner Sofa (Brown)
Category: Furniture
Sub-Category: Recliner Sofa
Material: Engineered wood frame with soft foam and thick webbing for back support
Upholstery: 260-gsm smooth nylon fabric in brown color
Comfort Features: Plush foam seating with webbing and springs for enhanced comfort
Usage: Ideal for reading, working, watching movies, or relaxing
Dimensions: 97 cm (W) x 97 cm (D) x 101 cm (H) | Weight: 39 kg
Price: ₹15,990 (66% off from MRP ₹47,900)"""
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

graph.add_node("get_product_info", context_manager_agent.get_product_info)
graph.add_node("intent_identifier", query_manager_agent.identify_intent)
graph.add_node("reformat_query", query_manager_agent.reformat_query)
graph.add_node("context_manager", context_manager_agent.context_process)  # 
graph.add_node("history_manager", history_manager_agent.history_process)
graph.add_node("action_manager", action_manager_agent.action_process)
graph.add_node("history_update", history_manager_agent.update_history)
graph.add_node("validate_response", validate_response_agent.validate_response_process)
graph.add_node("end", lambda state: state)  # This node just returns state as is

# Define Transitions
graph.set_entry_point("get_product_info")
graph.add_edge("get_product_info", "intent_identifier")
graph.add_edge("intent_identifier", "reformat_query")

# if query is not regarding simple product info use context manager else skip 
graph.add_conditional_edges(
    "reformat_query",
    lambda state: "context_manager" if state.intent == "Simple Product Info" else "proceed",
    {
        "context_manager": "context_manager",  # Skip to context manager
        "proceed": "action_manager",
    }
)

graph.add_edge("context_manager", "history_manager")
graph.add_edge("history_manager", "action_manager")
graph.add_edge("action_manager", "history_update")
graph.add_edge("action_manager", "validate_response")

graph.add_conditional_edges(
    "validate_response",
    lambda state: "query_manager" if state.retry_count == 1 else "end",
    {
        "query_manager": "intent_identifier",  # Retry once
        "end": "end"
    }
)

graph = graph.compile()
# endregion 

# Chatbot Interface
def chatbot(user_query, product_id, session_id, user_id):
    state = State(
        query=user_query, 
        root_product=product_id, 
        session_id=session_id,
        user_id=user_id
        )
    final_state = graph.invoke(state)
    return final_state.get("response", "No response generated.")


# Example Run
if __name__ == "__main__":
    # user_query = "How old is nilkamal ?"
    # user_query = "What is the warranty period?"
    # user_query = "Does it come with a power cord ?"
    # user_query = "How long is the power cord ? "
    # user_query = "Oh I thought it was an automatic one"
    # user_query = "Does it come in lighter colors? "
    # user_query = "Do you sell weather-resistant outdoor furniture?"
    queries = [
    "What are the key features of a high-quality recliner?",
    "How much space is needed behind a recliner for full extension?",
    "Do you have power recliners?",
    "Can I use a recliner sofa without plugging it in?",
    "Do recliner sofas require special flooring for stability?",
    "What are the different types of recliner sofas available?",
    "Which one do you offer?",
    "Do you offer single-seater, two-seater, and three-seater recliners?",
    "Show me 2-seater recliners",
    "Are recliner sofas available in sectional designs?",
    "How do I contact customer care?",
    "Do fabric recliner sofas stain easily?",
    "Care instructions",
    "What kind of foam or padding is used in the seating?",
    "How many reclining positions do your sofas offer?",
    "Which recliner do you recommend for a person with back pain?",
    "Do recliner sofas make noise when adjusting positions?"
]

    for user_query in queries:
        session_id = "trial_1234" 
        user_id = "Divya"
        # product_id = "Nilkamal Platinum Plastic Arm Chair (Season Rust Brown)"
        product_id = "Nilkamal Sierra 1 Seater Manual Recliner Sofa (Brown)"
        response = chatbot(
            user_query=user_query, 
            product_id=product_id, 
            session_id=session_id,
            user_id=user_id
            )
        print("Bot:", response)