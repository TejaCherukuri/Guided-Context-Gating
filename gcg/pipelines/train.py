'''
This file is used for end to end training from scratch and then, evaluation.
'''

from gcg.components import load_data, build_model, train_model, evaluate_model
from gcg import config
from gcg.utils import logging

logging.info("Initiated train pipeline")

logging.info(f"Loading data from {config.data_path}")
X_train, X_test, y_train, y_test = load_data(config.data_path, config.image_size)

logging.info("Building model...")
model = build_model(input_shape=config.image_size, num_classes=7)

logging.info("Training the model...")
train_model(model, X_train, X_test, y_train, y_test)

logging.info("Evaluating the model on test set...")
evaluate_model(model, X_test, y_test)

