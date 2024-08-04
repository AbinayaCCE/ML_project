import os
import sys
import dill
from src.exception import CustomException
from src.logger import logging

def save_obj(file_path, obj):
    try:
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved at {file_path}")
    except Exception as e:
        logging.error(f"Failed to save object at {file_path}: {e}")
        raise CustomException(e, sys)
