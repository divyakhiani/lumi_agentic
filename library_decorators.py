# library_decorators.py

# region IMPORT STANDARD LIBRARIES
import csv
from datetime import datetime
from functools import wraps
import inspect
import os
# endregion

# region IMPORT THIRD-PARTY LIBRARIES
import pandas as pd
import tiktoken
# endregion

# region IMPORT CONFIG LIBRARIES
from config_logger import logger
# endregion

# region FUNCTION: log_sample
def log_sample(value):
    """Helper function to generate log-friendly samples of inputs."""
    if isinstance(value, pd.DataFrame):
        # Log first row and headers of DataFrame
        return value.head(1).to_dict(orient='records')
    elif isinstance(value, (list, tuple)):
        # Log the first item of a list or tuple
        return value[:1]
    elif isinstance(value, dict):
        # Log the first key-value pair in a dictionary
        first_key = next(iter(value), None)
        return {first_key: value[first_key]} if first_key else {}
    elif isinstance(value, str) and len(value) > 100:
        # Log only the first 100 characters of long strings
        return value[:100] + "..."
    else:
        # Log as is for other data types, including None
        return value
# endregion

# region FUNCTION: log_function
def log_function(func):
    """Decorator to log the start, finish, and any exceptions of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"START_FUNCTION: {func.__name__}")
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"Exception in {func.__name__}: {e}")
            raise
        logger.info(f"COMPLETE_FUNCTION: {func.__name__}")
        return result
    return wrapper
# endregion

# region FUNCTION: log_function_with_args
def log_function_with_args(func):
    """Decorator to log function start, input parameters, finish, and exceptions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Create a sample log of arguments
        sampled_args = [log_sample(arg) for arg in args]
        sampled_kwargs = {k: log_sample(v) for k, v in kwargs.items()}

        logger.info(f"START_FUNCTION: {func.__name__} | ARGS: {sampled_args} | KWARGS: {sampled_kwargs}")
        
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"Exception in {func.__name__}: {e}")
            raise
        logger.info(f"COMPLETE_FUNCTION: {func.__name__}")
        return result
    return wrapper
# endregion

# region FUNCTION: log_token_usage
def count_tokens(messages, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    total_tokens = sum(len(encoding.encode(msg["content"])) for msg in messages)
    return total_tokens

# Get the actual calling function, skipping wrapper functions
def get_real_caller():
    stack = inspect.stack()
    for frame in stack:
        func_name = frame.function
        if func_name not in ("wrapper", "decorator", "log_token_usage","get_real_caller"):  # Ignore wrappers
            return func_name
    return "Unknown"  # Fallback if no valid function is found

# Decorator to log token count and save to a CSV file with a timestamped name
def log_token_usage(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        date_str = datetime.now().strftime("%Y-%m-%d")  # Generate timestamp
        folder = "token_usage"
        os.makedirs(folder, exist_ok=True) 
        csv_file = os.path.join(folder, f"llm_usage_{date_str}.csv")

        # messages = args[0]  # Assuming messages is the first argument
        messages = kwargs.get("messages", args[0] if args else None)
        model = kwargs.get("model", "gpt-3.5-turbo")
        
        token_count = count_tokens(messages, model)
        log_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        parent_function = get_real_caller()
        
        # Write to CSV
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "model", "token_count", "parent_function"])
            writer.writerow([log_time, model, token_count, parent_function])

        return func(*args, **kwargs)
    return wrapper

# endregion 