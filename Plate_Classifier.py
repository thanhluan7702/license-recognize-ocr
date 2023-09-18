import tensorflow.keras 
from PIL import Image, ImageOps 
import numpy as np 
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2 
cam = cv2.VideoCapture(0)

import warnings 
warnings.filterwarnings('ignore')


# -------------------------------------------

def plate_recognizie():
    # Disable scientific notation for clarity
    np.set_printoptions(suppress = True)

    # Load the model 
    model = tensorflow.keras.models.load_model('imageclassifier.h5', compile = False)

    # Create the keras model
    data = np.ndarray(shape = (1,256,256,3), dtype = np.float32)

    # Replace this with the path to your image 
    image = Image.open('img_detect.jpg')

    # Resize the image to a 256x256 
    size = (256, 256)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)  

    # Turn the image into a numpy array 
    image_array = np.asarray(image)

    # Display the resized image 
    # image.show()

    # Normalize the image 
    normalized_image_array = (image_array.astype(np.float32) / 255.0)

    # Load the image into the array 
    data[0] = normalized_image_array
    prediction = model.predict(data)

    if prediction[0][0] >= 0.5:
        object = 'Domestic_Motor'
        probability = prediction[0][0]
        print ("probability = " + str(probability))
        print("Prediction = " + object)
    else:
        object = 'Others'
        probability = 1 - prediction[0][0]
        print ("probability = " + str(probability))
        print("Prediction = " + object)

# if __name__ == "__main__": 
#     plate_recognizie()

def capture_image(): 
    ret, frame = cam.read()
    cv2.imwrite('img_detect.jpg', frame)

while True: 
    capture_image()
    plate_recognizie() 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows() 
