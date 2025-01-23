import sys
import joblib
from gcg.utils import logging, CustomException

def load_from_checkpoint(model, model_path):
    try:
        model.load_weights(model_path)
        return model
    except Exception as e:
        raise CustomException(e, sys)
    
def save_object(file_path, object):
    try:
        logging.info("Serializing the object as pickle file")
        joblib.dump(object, file_path)
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        logging.info("Deserializing the pickle file as object")
        return joblib.load(file_path)
    except Exception as e:
        raise CustomException(e, sys)