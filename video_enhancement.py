import os 
import random
import tensorflow as tf
from tensorflow import keras
import numpy as np
from glob import glob
from tensorflow import data 
from tensorflow import image

# import plotly.express as px
import matplotlib.pyplot as plt
import cv2

# model = tf.keras.models.load_model("/home/lengocthanh/projects/lowlight-enhance-mirnet/")

def preprocess_frame(frame, image_size=(124, 124)):
    # Convert the frame from BGR to RGB 
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to a tensor
    image = tf.convert_to_tensor(frame_rgb, dtype=tf.float32)
    
    # Resize 
    image = tf.image.resize(image, image_size)
    
    # Normalize 
    image = image / 255.0
    
    return image

def calculate_average_brightness(frame):
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    average_brightness = np.mean(gray_frame)
    
    return average_brightness

def enhance_image(frame, model):
    average_brightness = calculate_average_brightness(frame)
    if average_brightness <= 10:
        # model = tf.keras.models.load_model("F:/machine_learning/lowlight-enhance-mirnet")
        # mosdel = tf.keras.models.load_model("/home/lengocthanh/projects/lowlight-enhance-mirnet/")
        frm = preprocess_frame(frame)
        expand_img = tf.expand_dims(frm, axis = 0)
        output = model.predict(expand_img)
        return output[0]
    else:
        return frame

