'''
This file is used only for evaluation without the need to train the model
i.e Loads the best model and evaluates on your test dataset
'''

from gcg.components import load_data, build_model, evaluate_model
from gcg.utils import load_from_checkpoint, logging
from gcg import config

logging.info("Initiated evaluation pipeline")

logging.info(f"Loading data from {config.DATA_PATH}")
X_train, X_test, y_train, y_test = load_data(config.DATA_PATH, config.IMAGE_SIZE)

logging.info("Building model...")
model = build_model(input_shape=config.IMAGE_SIZE, num_classes=config.NUM_CLASSES)

logging.info("Loading the model checkpoint...")
model = load_from_checkpoint(model, config.FROM_HF, config.MODEL_SAVE_PATH)

logging.info("Evaluating the model on test set...")
evaluate_model(model, X_test, y_test)
