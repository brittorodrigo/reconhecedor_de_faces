#!/usr/bin/python

import cv2, os
import numpy as np
import sys

# Utilizando Haar Cascade do OpenCV para detectar as faces.
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# Utilizaremos para criar o modelo da face que queremos reconhecer o
#  LBPH Face Recognizer
recognizer = cv2.createLBPHFaceRecognizer()


# Validacao do formato da imagem aceita; eh valido png,jpeg,jpg.
def is_valid_image(image_name):
    if (image_name.endswith('.jpg') or image_name.endswith('.jpeg') or image_name.endswith('.png')):
        return True
    return False


def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if not 'test' in f and is_valid_image(f)]
    # conjunto com as faces extraidas
    images = []
    # rotulo da imagem
    labels = []
    for image_path in image_paths:
        print image_path
        # Ler a imagem.
        imagem_de_entrada = cv2.imread(image_path)
        # Converter a imagem para tons de cinza
        imagem_tons_de_cinza = cv2.cvtColor(imagem_de_entrada, cv2.COLOR_BGR2GRAY)
        # Criando um Numpy Array
        imagem_arrayNumPY = np.array(imagem_tons_de_cinza, 'uint8')
        # Obtendo o rotulo da imagem, ID
        nbr = int(os.path.split(image_path)[1].split(".")[1])
        # Detectar a face na Imagem.
        faces = faceCascade.detectMultiScale(
            imagem_arrayNumPY,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
        # Se alguma face for encontrada, adicionar a face para Face e o rotulo(id) para o rotulo)
        for (x, y, w, h) in faces:
            images.append(imagem_arrayNumPY[y: y + h, x: x + w])
            labels.append(nbr)
            # cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
            cv2.waitKey(50)
    return images, labels


# Caminho do computador para o conjunto de treinamento
training_set = sys.argv[1]
rostos_previamente_verificados = sys.argv[2]

images, labels = get_images_and_labels(training_set)
cv2.destroyAllWindows()

# Realizar o treinamento baseado no conjunto de imagens fornecidas.
recognizer.train(images, np.array(labels))

path = './bdtest'
image_paths = [os.path.join(path, f) for f in os.listdir(path)]
faces_corretamente_reconhecidas = 0
faces_Incorretamente_reconhecidas = 0
for image_path in image_paths:
    face_a_ser_reconhecida = cv2.imread(image_path)
    face_a_ser_reconhecida_tons_cinza = cv2.cvtColor(face_a_ser_reconhecida, cv2.COLOR_BGR2GRAY)
    face_a_ser_reconhecida_numPyArray = np.array(face_a_ser_reconhecida_tons_cinza, 'uint8')
    faces = faceCascade.detectMultiScale(face_a_ser_reconhecida_numPyArray,
                                         scaleFactor=1.2,
                                         minNeighbors=5,
                                         minSize=(30, 30),
                                         flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
    for (x, y, w, h) in faces:
        nbr_predicted2, conf = recognizer.predict(face_a_ser_reconhecida_numPyArray[y: y + h, x: x + w])
        nbr_actual = int(os.path.split(image_path)[1].split(".")[1])
        # threshold = 100
        if nbr_actual == nbr_predicted2 and conf <= 90:
            true_confidence = 100 - conf;
            print "{} eh corretamente reconhecido com nivel de confianca {}".format(nbr_actual, true_confidence)
            cv2.rectangle(face_a_ser_reconhecida, (x, y), (x + w, y + h), (0, 255, 0), 2)
            faces_corretamente_reconhecidas += 1
            cv2.imshow("Face reconhecida", face_a_ser_reconhecida)
        elif nbr_actual != nbr_predicted2 and conf <= 10:
            print "{} eh incorretamente reconhecido como {} com nivel de confianca {}".format(nbr_actual,
                                                                                              nbr_predicted2,
                                                                                              100 - conf)
            print conf
            faces_Incorretamente_reconhecidas += 1
            cv2.rectangle(face_a_ser_reconhecida, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("Face incorretamente reconhecida", face_a_ser_reconhecida)
        cv2.waitKey(1000)
print "--X--X--X--X--X--X--X--X--X--X--"
print "Acerto(s): {}\nErro(s) ou (falso positivo): {}".format(faces_corretamente_reconhecidas,
                                                      faces_Incorretamente_reconhecidas)
print "Rostos nao reconhecidos e que eram pra ser ou (falso negativo): {}".format(
    int(rostos_previamente_verificados) - faces_corretamente_reconhecidas)
if((faces_Incorretamente_reconhecidas+faces_corretamente_reconhecidas) >0):
    print "Precisao: {}".format(faces_corretamente_reconhecidas / (faces_corretamente_reconhecidas + faces_Incorretamente_reconhecidas))
log_path = os.path.join(training_set, training_set + "_log.txt")
log = open(log_path, "w")
log.write("acertos\n" + str(faces_corretamente_reconhecidas))
log.write("\nerros\n" + str(faces_Incorretamente_reconhecidas))
log.close()
