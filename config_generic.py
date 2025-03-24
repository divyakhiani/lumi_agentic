# region IMPORT STANDARD LIBRARIES
import os
import sys
import json
import shutil
import logging

from functools import wraps
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
# endregion

# region FUNCTION: create_folder
def create_folder(dir_folder):
    # Check if the folder exists
    if not os.path.exists(dir_folder):
        # If it doesn't exist, create it
        os.makedirs(dir_folder)
    else:
        output = None
# endregion 

# region FUNCTION: create_folder
def create_folder(dir_folder, delete_if_exists=False):
    """
    Creates a folder. If the folder exists, it can optionally be deleted and recreated.

    Args:
        dir_folder (str): The path of the folder to create.
        delete_if_exists (bool): If True, deletes the folder and its contents if it already exists.
    """
    if os.path.exists(dir_folder):
        if delete_if_exists:
            # Delete the folder and its contents recursively
            shutil.rmtree(dir_folder)
        else:
            return
    # Create the new folder
    os.makedirs(dir_folder)
# endregion

# region SCRIPT CONFIGURATION
path_prj = os.getcwd()
# endregion 

# region CLASS: GenericConfig
class GenericConfig:
    def __init__(self, env=os.getenv('ENVIRONMENT', 'production')):

        # region ENVIRONMENT INDEPENDENT CONFIGURATION
        self.env = env
        self.path_prj = path_prj

        self.dir_ip = os.path.join(self.path_prj, "input")
        create_folder(self.dir_ip, delete_if_exists=False)

        self.dir_op = os.path.join(self.path_prj, "output")
        create_folder(self.dir_op, delete_if_exists=False)

        self.dir_query = self.dir_ip

        self.dir_log = os.path.join(self.path_prj, "logs")
        create_folder(self.dir_log, delete_if_exists=False)
        
        self.dir_support = os.path.join(self.path_prj, "support")
        self.dir_session = os.path.join(self.path_prj, "session")
        self.dir_database = os.path.join(self.path_prj, "database")
        create_folder(self.dir_session, delete_if_exists=False)

        self.file_prj_query = "sql_queries.yaml"
        self.file_model_config = "model_config.json"
        self.prompts = "prompts.yml" 
        # endregion

        # region RELATIONAL DATABASE
        self.oracle = {}
        self.postgres = {}        
        self.sql_server = {}
        self.schemas = {}
        self.tables = {}
        # endregion

        # region NON RELATIONAL DATABASE
        self.mongo = {}
        # endregion

        # region AWS CONFIGURATION
        # endregion
 
        # region READING ENVIRONMENT SPECIFIC CONFIGURATION & CREDENTIAL
        with open(os.path.join(path_prj, "credentials", "credentials.json")) as file:
            dict_env = json.load(file)
        # endregion

        # region EXTERNAL SYSTEM APIS
        self.api_keys = {}
        self.api_env = {}
        self.api_headers = {}
        self.models = {}
        self.users = {}
        # endregion

        # region ENVIRONMENT DEPENDENT CONFIGURATION: local
        self.postgres.update(dict_env.get(self.env, {}).get('postgres', {}))
        self.oracle.update(dict_env.get(self.env, {}).get('oracle', {}))
        self.sql_server.update(dict_env.get(self.env, {}).get('sql_server', {}))
        self.mongo.update(dict_env.get(self.env, {}).get('mongo', {}))
        self.api_env.update(dict_env.get(self.env, {}).get('api_env', {}))
        self.models.update(dict_env.get(self.env, {}).get('models', {}))
        self.users = dict_env.get(self.env, {}).get('users', [])
        self.collection_chroma = "nilkamal_row_embeddings_ext"
        self.chosen_model = 'openai'
        # endregion
        
        # region READING API CONFIGURATION
        # endregion

        # region APIS CONFIGURATION
        # endregion

        # region FILES PRODUCTS AND REVIEWS
        self.file_name_products = os.path.join(self.dir_op, "nilkamal/nilkamal_products_data.csv")
        self.file_name_reviews = os.path.join(self.dir_op, "nilkamal/nilkamal_customer_reviews.csv")
        # endregion

        # region DECLARE ADDITIONAL CONFIGURATION FOR ENVIRONMENTS
        if self.env in ['local']:
            pass

        if self.env in ['staging']:
            pass

        if self.env in ['production']:
            pass
        # endregion
        
# endregion

# region MODULE CLASS: Self
class self():
    def __init__(self, env=os.getenv('ENVIRONMENT')):
        super().__init__(env)       
# endregion
