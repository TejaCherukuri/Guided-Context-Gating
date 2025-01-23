import os
import cv2
import numpy as np
import sys
from gcg import config
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from gcg.utils import logging, CustomException, save_object


def preprocess_image(img_path, image_size):
    # Read the image from the specified path
    img = cv2.imread(img_path)
    # Convert the image from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array=cv2.resize(img, (image_size[0],image_size[1]), fx=1, fy=1,interpolation = cv2.INTER_CUBIC)  

    return img_array

# Data Loading
def load_data(data_path, image_size):
    try:
        subfolders = config.labels
        logging.info("Dataset Loading...")

        img_data_list=[]
        labels_list = []
        num_images_per_class = []

        for category in subfolders:
            img_list=os.listdir(data_path +'/'+ category)
            if("Annotations" in category):
                continue
            
            logging.info(f'Loading : {len(img_list)}, images of category: {category}')
            for img in img_list:
                # Load an image from this path
                img_path = data_path + '/'+ category + '/'+ img

                # Preprocess image
                img_array=preprocess_image(img_path, image_size)  

                img_data_list.append(img_array) 
                labels_list.append(category)
            num_images_per_class.append(len(img_list))

        le = LabelEncoder()
        labels = le.fit_transform(labels_list)
        labels = to_categorical(labels)

        # Saving the label encoder object for use during inference
        save_object(config.labelencoder_save_path, le)

        data = np.array(img_data_list)

        # Dataset Summary
        logging.info(f"Total number of uploaded data: {data.shape[0]} with data shape, ({data.shape[1]},{data.shape[2]},{data.shape[3]})")
        
        logging.info("Initiated train_test_split")
        X_train, X_test, y_train, y_test = initiate_train_test_split(data, labels)

        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        raise CustomException(e, sys)

# Train Test Split
def initiate_train_test_split(data, labels):

    # Split the dataset into two subsets (80%-20%). The first one will be used for training.
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=195, stratify=labels)

    logging.info(f"X_train has shape: {X_train.shape}")
    logging.info(f"y_train has shape: {y_train.shape}\n")

    logging.info(f"X_test has shape: {X_test.shape}")
    logging.info(f"y_test has shape: {y_test.shape}\n")

    logging.info(f"X_train + X_test = {X_train.shape[0] + X_test.shape[0]} samples in total")

    return X_train, X_test, y_train, y_test