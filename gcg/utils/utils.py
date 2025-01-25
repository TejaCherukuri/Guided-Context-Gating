from huggingface_hub import hf_hub_download
import sys
import joblib
from gcg import config
from gcg.utils import logging, CustomException

def load_from_checkpoint(model, model_path, from_hf):
    logging.info("Loading the model from checkpoint...")
    try:
        if from_hf:
            logging.info("From huggingface...")
            # Download the file from Hugging Face Model Hub
            local_path = hf_hub_download(repo_id=config.REPO_ID, filename=config.WEIGHT_FILE)
            model.load_weights(local_path)
        else:
            # Local path
            model.load_weights(model_path)
        return model
    except Exception as e:
        raise CustomException(e, sys)
    
def save_object(file_path, object):
    logging.info("Serializing the object as pickle file")
    try:
        joblib.dump(object, file_path)
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path, from_hf):
    logging.info("Deserializing the pickle file as object")
    try:
        if from_hf:
            logging.info("From huggingface...")
            local_path = hf_hub_download(repo_id=config.REPO_ID, filename=config.LABELENCODER_FILE)
            return joblib.load(local_path)
        else:
            return joblib.load(file_path)
    except Exception as e:
        raise CustomException(e, sys)