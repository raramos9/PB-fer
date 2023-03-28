
# TODO - Import Libraries 
import time
import uuid 
import cv2
import os




# TODO - Image Labels




# TODO - Folder Setup / Training and Testing Folders

print(cv2.__version__)

IMAGE_PATH = os.path.join( 'Tensorflow', 'workspace', 'images', 'collectedimages')
if not os.path. exists (IMAGE_PATH):
        os.makedirs (IMAGE_PATH)
labels = ['anger', 'fear', 'disgust', 'happy' , 'sad' , 'neutral' , 'suprised']
num_img = 1
for label in labels:
    path = os.path. join (IMAGE_PATH, label)
    if not os.path.exists (path): 
            os.makedirs (path)






# TODO - Capture Images




# TODO - Labeling Images 