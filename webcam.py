import cv2
import numpy as np

#Gera um classificador
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
#inicia a webcam
captura = cv2.VideoCapture(0)

while(True):
	ret, frame = captura.read()
	
	#Filtro escala de cizas
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#Detecta a face usando o classificador
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
	eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
	smiles = smile_cascade.detectMultiScale(gray, minSize=(55,55), minNeighbors=30, scaleFactor=1.7)
	#desenha os retangulos
	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x,y), (x+w, y+h), (125,225,0), 2)
	for (x, y, w, h) in eyes:
		cv2.rectangle(frame, (x,y), (x+w, y+h), (255,225,0), 2)
	for (x, y, w, h) in smiles:
		cv2.rectangle(frame, (x,y), (x+w, y+h), (10,0,100), 2)
	#exibe o video
	cv2.imshow('Video', frame)
	#Aguarda pressionar ESC para sair
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break
#Limpa o cache
captura.release()
#Fecha todas as janelas
cv2.destroyAllWindows()