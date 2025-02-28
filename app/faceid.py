
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np

# Building app and layout
class CamApp(App):
    def build(self):
        self.web_cam = Image(size_hint = (1,.8))
        self.button = Button(text = "Verify", on_press = self.verify, size_hint = (1,.1))
        self.verification_label = Label(text='Verification Uninitiated', size_hint = (1,.1))

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        self.model = tf.keras.models.load_model('/Users/AshishR_T/Desktop/Timepass python projects/Deep Learning Model/app/siameseModelV2.h5', custom_objects={'L1Dist':L1Dist})

        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)

        return layout
    
    def update(self, *args):
        ret, frame = self.capture.read()

        if not ret:  # If frame is not valid, skip and continue looping
            print("Failed to capture frame, retrying...")
            return  # Skip this iteration of the update method

        frame = frame[250:500, 600:850, :]  # Crop the frame if needed

        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture
    
    # PreProcessing - Scaling and Resizing
    def preprocess (self, file_path):
        byte_img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(byte_img)
        img = tf.image.resize(img, (105,105))
        img = img / 255.0
        return img

    # Verification function
    def verify(self, *args):
        detection_threshold = 0.5
        verification_threshold = 0.8

        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()

        if not ret:  # If frame is not valid, skip and continue looping
            print("Failed to capture frame, retrying...")
            return  # Skip this iteration of the update method

        frame = frame[250:500, 600:850, :]
        cv2.imwrite(SAVE_PATH, frame)

        # Build results array
        results = []
        # Looping through the verification images in the application_data folder
        for image in os.listdir(os.path.join('application_data', 'verification_images')):
            if not image.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # Ignore non-images
                continue
            # Assigning input_img input_image.jpg
            input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
            # Preprocessing each image and assigning it to validation_img 
            validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))

            # Making predictions
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)

        # Detection Threshold: Metric above which a prediction is considered positive
        detection = np.sum(np.array(results) > detection_threshold)

        # Verification Threshold: Proportion of positive predictions / total positive samples
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
        verified = verification > verification_threshold

        self.verification_label.text = 'Verified' if verified == True else 'Unverified'

        Logger.info(results)
        Logger.info(np.sum(np.array(results)>0.4))
        Logger.info(np.sum(np.array(results)>0.5))
        Logger.info(np.sum(np.array(results)>0.8))
        Logger.info(np.sum(np.array(results)>0.9))

        print()
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified)

        return results, verified
    

if __name__ == "__main__":
    CamApp().run()