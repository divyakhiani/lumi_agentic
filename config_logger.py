# region IMPORT STANDARD LIBRARIES
import os
import logging

from datetime import datetime
# endregion

# region IMPORT CONFIG LIBRARIES
from config_generic import GenericConfig as Config 
# endregion

# region LOAD CONFIG
config = Config()
# endregion

# region LOGGER CONFIGURATION
now = datetime.now()
today = now.strftime('%Y%m%d')
log_filename = config.dir_log + f"/{today}_logs.log"
logger = logging.getLogger("central_logger")
logger.setLevel(logging.DEBUG)
# endregion

# region LOGGER HANDLER CONFIFURATION
if not logger.hasHandlers():

    # Create a file handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)

    # Create a console handler to stream logs to stdout (optional, for debugging)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Define a formatter and add it to handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# endregion
