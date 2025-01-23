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

def predict(img_paths:List):
    '''
    Inputs: List of image paths
    Output: Predictions List and Generates heatmaps
    '''
    try:
        predictions_list = []
        # Step 1: Building the model
        model = build_model(input_shape=config.image_size, num_classes=config.num_classes)

        # Step 2: Loading the model from checkpoint     
        logging.info("Loading the model from checkpoint...")
        model = load_from_checkpoint(model, config.model_path)

        # Step 3: Loading the label encoder to decode indices
        le = load_object(config.labelencoder_save_path)

        for img_path in img_paths:
            # Step 4: Read and preprocess the image
            img_name = img_path.split("/")[-1]

            resized_img = preprocess_image(img_path, config.image_size)
            img_array = np.expand_dims(resized_img, axis=0)
            
            # Step 5: Inference on the model
            logging.info("Getting your prediction...")
            pred = model.predict(img_array)
            predicted_class = np.argmax(pred, axis=1)[0]  # Get class index
            
            logging.info(f"Prediction: {le.classes_[predicted_class]}, Path: {img_name}")
            predictions_list.append(le.classes_[predicted_class])
            
            # Step 6: Generating heatmap for the image
            logging.info("Generating the heatmap using GradCAM++")
            heatmap_plus = grad_cam_plus(model, resized_img, config.gcg_layer_name, label_name=config.labels, category_id=predicted_class)

            os.makedirs(config.heatmaps_save_path, exist_ok=True) 
            heatmap_img = config.heatmaps_save_path + f'/heatmap_{img_name}'
            show_GradCAM(resized_img, heatmap_plus, save_path=heatmap_img)

        return predictions_list

    except Exception as e:
        raise CustomException(e, sys)
    
if __name__=='__main__':
    predictions_list = predict(config.test_images)
    print(predictions_list)