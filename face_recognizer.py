#!/usr/bin/python

# Import the required modules
import cv2, os
import numpy as np
#from PIL import Image

# For face detection we will use the Haar Cascade provided by OpenCV.
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# For face recognition we will the the LBPH Face Recognizer
recognizer = cv2.createLBPHFaceRecognizer()



def get_images_and_labels(path):
    # Append all the absolute image paths in a list image_paths
    # We will not read the image with the .sad extension in the training set
    # Rather, we will use them to test our accuracy of the training
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if not 'test' in f and f.endswith('.jpg')]
    # images will contains face images
    images = []
    # labels will contains the label that is assigned to the image
    labels = []
    for image_path in image_paths:
	print image_path
        # Read the image and convert to grayscale
        #image_pil = Image.open(image_path).convert('L')		
	input_image = cv2.imread(image_path)
        image_pil = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        # Convert the image format into numpy array
        image = np.array(image_pil, 'uint8')
        # Get the label of the image
        #nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
	nbr = int(os.path.split(image_path)[1].split(".")[1])
        # Detect the face in the image
        faces = faceCascade.detectMultiScale(image)
        # If face is detected, append the face to images and the label to labels
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(nbr)
            cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
            cv2.waitKey(50)
    # return the images list and labels list
    return images, labels

# Path to the Yale Dataset
path = './bdangelina'
# Call the get_images_and_labels function and get the face images and the
# corresponding labels
images, labels = get_images_and_labels(path)
cv2.destroyAllWindows()

# Perform the tranining
recognizer.train(images, np.array(labels))

# Append the images with the extension .sad into image_paths
path = './bdtest'			
image_paths = [os.path.join(path, f) for f in os.listdir(path)]
correct_matches = 0
incorrect_matches = 0
for image_path in image_paths:
    #predict_image_pil = Image.open(image_path).convert('L')
    input_predict_image = cv2.imread(image_path)
    predict_image_pil = cv2.cvtColor(input_predict_image, cv2.COLOR_BGR2GRAY)
    predict_image = np.array(predict_image_pil, 'uint8')
    faces = faceCascade.detectMultiScale(predict_image)
    for (x, y, w, h) in faces:
        nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
        nbr_actual = int(os.path.split(image_path)[1].split(".")[1])
	threshold = 100
        if nbr_actual == nbr_predicted  and conf < 100:
	    true_confidence = 100 - conf;
            print "{} eh corretamente reconhecido com nivel de confianca {}".format(nbr_actual, true_confidence)
	    cv2.rectangle(input_predict_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
	    correct_matches += 1
	    cv2.imshow("Face reconhecida" ,input_predict_image)
        elif nbr_actual != nbr_predicted  and  conf > 2 and conf < 100:
            print "{} eh incorretamente reconhecido como {}".format(nbr_actual, nbr_predicted)
	    correct_matches += 1
        cv2.waitKey(1000)
print "acertos: {}\nerros: {}".format(correct_matches, incorrect_matches)
