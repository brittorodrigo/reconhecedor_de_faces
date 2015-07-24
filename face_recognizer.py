#!/usr/bin/python

import cv2, os
import numpy as np
import sys
import matplotlib.pyplot as pyplot



# Validacao do formato da imagem aceita; eh valido png,jpeg,jpg.
def eh_imagem_valida(image_name):
    if (image_name.endswith('.jpg') or image_name.endswith('.jpeg') or image_name.endswith('.png')):
        return True
    return False



def recortar_face(imagem_array, x, y, w, h):
	# redimensionar face se usar o Eigen Recognizer
	if(recognizer_id == 2):
		face_temp = imagem_array[y:y + h, x:x + w]
		face = cv2.resize(face_temp, (200, 200))
	else:			
	    	face = imagem_array[y: y + h, x: x + w]
	return face


def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if not 'test' in f and eh_imagem_valida(f)]
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
        imagem_array = np.array(imagem_tons_de_cinza, 'uint8')
        # Obtendo o rotulo da image(ID)
        rotulo = int(os.path.split(image_path)[1].split(".")[1])
        # Detectar a face na Imagem.
        faces = faceCascade.detectMultiScale(
            imagem_array,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
        # Se alguma face for encontrada, guardar a face e seu respectivo rotulo (id)
        for (x, y, w, h) in faces:
	     #Para Eigen Recognizer e preciso redimensionar as faces para o mesmo tamanho 
	    face = recortar_face(imagem_array,x,y,w,h)
            images.append(face)
            labels.append(rotulo)
            # cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
            cv2.waitKey(50)
    return images, labels

def escolher_recognizer(parametro):
	if(parametro == 1):
		return cv2.createLBPHFaceRecognizer()
	elif(parametro == 2):
		return cv2.createEigenFaceRecognizer()				




# Utilizando Haar Cascade do OpenCV para detectar as faces.
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

	
# Caminho do diretorio do conjunto de treinamento
training_set = sys.argv[1]
faces_reconhecidas_esperadas = sys.argv[2]

# Utilizaremos o LBPH Face ou EigenFace Recognizer, para criar o modelo da face que queremos reconhecer 
recognizer_id = int(sys.argv[3])
recognizer = escolher_recognizer(recognizer_id)
	
images, labels = get_images_and_labels(training_set)
cv2.destroyAllWindows()

# Realizar o treinamento baseado no conjunto de faces fornecidas.
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
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(30, 30),
                                         flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
    for (x, y, w, h) in faces:
	#Para Eigen Recognizer e preciso redimensionar as faces para o mesmo tamanho 
	face = recortar_face(face_a_ser_reconhecida_numPyArray,x,y,w,h)
	#cv2.imshow("face recortada", face)
        rotulo_classificado, conf = recognizer.predict(face)
        rotulo_real = int(os.path.split(image_path)[1].split(".")[1])
        if rotulo_real == rotulo_classificado and conf <= 90:
            true_confidence = 100 - conf;
            print "{} eh corretamente reconhecido com nivel de confianca {}".format(rotulo_real, true_confidence)
            cv2.rectangle(face_a_ser_reconhecida, (x, y), (x + w, y + h), (0, 255, 0), 2)
            faces_corretamente_reconhecidas += 1
            cv2.imshow("Face reconhecida", face_a_ser_reconhecida)
        elif rotulo_real != rotulo_classificado and conf <= 10:
            print "{} eh incorretamente reconhecido como {} com nivel de confianca {}".format(rotulo_real,
                                                                                              rotulo_classificado,
                                                                                              100 - conf)
            faces_Incorretamente_reconhecidas += 1
            cv2.rectangle(face_a_ser_reconhecida, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("Face incorretamente reconhecida", face_a_ser_reconhecida)
        cv2.waitKey(1000)

falsos_negativos = int(faces_reconhecidas_esperadas) - faces_corretamente_reconhecidas
print "--X--X--X--X--X--X--X--X--X--X--"
print "Acerto(s): {}\nErro(s) ou (falso positivo): {}".format(faces_corretamente_reconhecidas,faces_Incorretamente_reconhecidas)
print "Rostos nao reconhecidos e que eram pra ser ou (falso negativo): {}".format(falsos_negativos)
fscore = 0
if ((faces_Incorretamente_reconhecidas + faces_corretamente_reconhecidas) > 0):

    precisao =  (faces_corretamente_reconhecidas+0.0) / (faces_corretamente_reconhecidas + faces_Incorretamente_reconhecidas)
    print "Precisao: {}".format(precisao)
    recall = (faces_corretamente_reconhecidas+0.0)/(faces_corretamente_reconhecidas + falsos_negativos)
    print "Recall: {}".format(recall)

    fscore = 2.0*((precisao * recall)/(precisao + recall))
    print "F1 SCORE: {}".format(fscore)

log_path = os.path.join(training_set, training_set + "_log.txt")
log = open(log_path, "w")
log.write("acertos\n" + str(faces_corretamente_reconhecidas))
log.write("\nerros\n" + str(faces_Incorretamente_reconhecidas))
log.close()

x_list = [fscore]
label_list = ["F1 Score"]
pyplot.subplot(2, 1, 1)

pyplot.axis("equal")
pyplot.pie(
    x_list,
    labels=label_list,
    autopct="%1.1f%%"
)
pyplot.title("Visao Geral")
y = [faces_corretamente_reconhecidas, faces_Incorretamente_reconhecidas,
          int(faces_reconhecidas_esperadas) - faces_corretamente_reconhecidas]
N = len(y)
x = range(N)
width = 1/1.5
pyplot.subplot(2, 1, 2)

pyplot.bar(x, y, width, color="blue",align='center')
pyplot.xticks(x,["Acertos","Falsos Positivos","Falsos Negativos"])
pyplot.show()
