'''
    This file is used for inference on new test samples.
    One sample or mutiple samples can be fed to the predict() method
'''
import os
import sys
from typing import List
import numpy as np
from gcg import config
from gcg.utils import logging, CustomException, load_from_checkpoint, load_object
from gcg.components import build_model, preprocess_image, grad_cam_plus, show_GradCAM

def predict(img_paths:List[str]):
    '''
    Inputs: List of image paths
    Output: Predictions List and Generates heatmaps
    '''
    try:
        predictions = {}
        # Step 1: Building the model
        model = build_model(input_shape=config.IMAGE_SIZE, num_classes=config.NUM_CLASSES)

        # Step 2: Loading the model from checkpoint     
        model = load_from_checkpoint(model, config.MODEL_SAVE_PATH, config.FROM_HF)

        # Step 3: Loading the label encoder to decode indices
        le = load_object(config.LABELENCODER_SAVE_PATH, config.FROM_HF)

        for img_path in img_paths:
            # Step 4: Read and preprocess the image
            img_name = img_path.split("/")[-1]

            resized_img = preprocess_image(img_path, config.IMAGE_SIZE)
            img_array = np.expand_dims(resized_img, axis=0)
            
            # Step 5: Inference on the model
            logging.info("Getting your prediction...")
            pred = model.predict(img_array)
            predicted_class = np.argmax(pred, axis=1)[0]  # Get class index
            
            logging.info(f"Prediction: {le.classes_[predicted_class]}, Path: {img_name}")
            predictions[img_name] = le.classes_[predicted_class].item()
            
            # Step 6: Generating heatmap for the image
            logging.info("Generating the heatmap using GradCAM++")
            heatmap_plus = grad_cam_plus(model, resized_img, config.GCG_LAYER_OUTPUT, label_name=config.LABELS, category_id=predicted_class)

            os.makedirs(config.HEATMAPS_SAVE_PATH, exist_ok=True) 
            heatmap_img = config.HEATMAPS_SAVE_PATH + f'/heatmap_{img_name}'
            show_GradCAM(resized_img, heatmap_plus, save_path=heatmap_img)

        return predictions

    except Exception as e:
        raise CustomException(e, sys)
    
if __name__=='__main__':
    predictions = predict(config.TEST_IMAGES)
    print(predictions)