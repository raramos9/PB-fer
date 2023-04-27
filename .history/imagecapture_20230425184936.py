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

for label in labels:
       cap = cv2.VideoCapture(0)

       print("Collecting image(s) for {}".format(label))

       for img_num in range(num_img): #captures number of images specified
                print("Collecting image {} in ".format(img_num))
                for count in range(3, 0, -1): #countdown timer
                       print(count)
                       time.sleep(1)
                
                ret, frame = cap.read()
                imgname = os.path.join(IMAGE_PATH, label, label + '.' + '{}.jpg'.format(str(uuid.uuid1()))) #assigning id and making jpg file

                cv2.imwrite(imgname, frame)
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                       break

cap.release()
cv2.destroyAllWindows()

# TODO - Labeling Images 



