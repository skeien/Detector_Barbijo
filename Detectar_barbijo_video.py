# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
	# Marcar las dimensiones de la trama de datos y construir el modelo
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# Pasar el modelo a través de la red y obtener las detecciones de rostros
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# inicializar nuestra lista de caras, sus ubicaciones correspondientes,
	# y la lista de predicciones de nuestra red de mascarillas
	faces = []
	locs = []
	preds = []

	# loop para detectar
	for i in range(0, detections.shape[2]):
		# extraer la probabilidad(confidence) asociada con
		# la deteccion
		confidence = detections[0, 0, i, 2]

		# Filtrar las detecciones debiles y fortalecer las probabilidades
		# mas importantes de las menos importantes
		if confidence > 0.5:
			# calcular las coordenadas (x, y) del cuadro delimitador para
			# formar el objeto
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# Asegúrese de que los cuadros delimitadores caigan dentro de las dimensiones de
			# la trama
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extraer el ROI de la cara, convertirlo de BGR a canal RGB
			# realizar el pedido, cambiar su tamaño a 224x224 y preprocesarlo
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# agregue la cara y los cuadros delimitadores a sus respectivas listas
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# Solo haga predicciones de "if" se detectó al menos una cara
	if len(faces) > 0:
		# Para una inferencia más rápida, haremos predicciones por lotes en * todas *
		# las caras al mismo tiempo en lugar de predicciones una por una
		# en el bucle "for" anterior
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# devuelve un tuple de 2 de las ubicaciones de las caras y sus 
	# correspondientes localizaciones
	return (locs, preds)

# Cargar nuestro modelo de detector facial serializado desde el disco
from os.path import dirname, join

prototxtPath = join(dirname(__file__), "face_detector\deploy.prototxt")
weightsPath = join(dirname(__file__), "face_detector\res10_300x300_ssd_iter_140000.caffemodel")

"""
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
"""

faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# cargar el modelo del detector de mascarilla desde el disco
maskNet = load_model("mask_detector.model")

# inicializar el video stream
print("[CdR] Iniciando Video Stream...")
vs = VideoStream(src=0).start()

# Loop de los fotogramas de la secuencia de vídeo
while True:
	# Tomar el fotograma de la secuencia de video estudiada y cambie su tamaño 
	# para que tenga un ancho máximo de 400 píxeles
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# Detectar rostros en el marco y determinar si están usando un
	# Barbijo o no
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# recorrer las ubicaciones de las caras detectadas y sus correspondientes
	# lugares
	for (box, pred) in zip(locs, preds):
		# Desempaquetar el cuadro delimitador y las predicciones
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# determinar la etiqueta de clase y el color que usaremos para dibujar
		# el cuadro de contorno y el texto
		label = "Barbijo" if mask > withoutMask else "Sin Barbijo"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# incluir la probabilidad
		label = "{}: {:.1f}%".format(label, max(mask, withoutMask) * 100)

		# Mostrar la etiqueta y el rectángulo del cuadro delimitador en la salida
		# de la trama
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# Mostrar el marco de salida
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# Si se presioná la tecla `Q`, sale del bucle
	if key == ord("q"):
		break

# Limpieza liviana del disco
cv2.destroyAllWindows()
vs.stop()
