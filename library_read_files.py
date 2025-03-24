# library_read_files 

# region IMPORT STANDARD LIBRARIES
import os
import re
import json
import groq
import openai
import pandas as pd
import yaml
# endregion

# region IMPORT THIRD-PARTY LIBRARIES

# endregion 

# region IMPORT CUSTOM LIBRARIES
import library_decorators as libdec
# endregion

# region IMPORT CONFIG LIBRARIES
from config_generic import GenericConfig as Config
from config_logger import logger
# endregion 

# region LOAD CONFIG, API
config = Config()
# endregion

# region FUNCTION read_prompts
def read_prompts(file_path):
    with open(file_path, "r") as file:
        prompts = yaml.safe_load(file)

    return prompts
# endregion 

# region FUNCTION: read_data
@libdec.log_function
def read_data(file):
    if file:
        logger.info(f"READING_FILE: {file}")
    else:
        logger.info(f"EMPTY_FILE: {file}")

    try:        
        data = pd.read_csv(file)
        logger.info(f"READING_SUCCESSFUL: SHAPE_OF_DATA: {data.shape}")
        return data   
    except Exception as e:
        logger.error(f"READING_FAILED: {file}, RETURNING_EMPTY_DATAFRAME")
        return pd.DataFrame()
# endregion

# region FUNCTION: get_completion_from_llm
@libdec.log_function
@libdec.log_token_usage
def get_completion_from_llm(
    messages, 
    model="gpt-3.5-turbo", 
    temperature=0, 
    max_tokens=500
):
    try:
        if config.chosen_model=="openai":
            logger.info("CALLING_OPENAI_API: ")
            openai.api_key = config.models['openai']['key_001']
            response = openai.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            logger.info(f"SUCCESS_IN_EXECUTE_LLM_API")
            return response.choices[0].message.content
        else:
            logger.info("USING_GROQ_API: ")
            groq_client = groq.Client(api_key=config.models['groq']['key_002'])

            response = groq_client.chat.completions.create(
                model=config.models['groq']['model_001'],
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            logger.info(f"SUCCESS_IN_EXECUTE_LLM_API")
            return response.choices[0].message.content
    except Exception as e:
        logger.error(f"FAILED_TO_EXECUTE_LLM_API: with error: {e}")
        return None

# endregion 