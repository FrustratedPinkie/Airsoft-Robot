from keras.models import load_model
import cv2
import numpy as np
import requests

model = load_model('model-020.model')

face_class = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
source = cv2.VideoCapture(0)

labels_dict={0: 'Angry', 1: 'Happy', 2:'Neutral'}
color_dict={0:(0,0,255), 1:(0,255,0), 2:(255,0,0)}

trigger = 25
counter = 0

while(True):

    ret, img = source.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_class.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        face_img = gray[y: y + w, x: x + w]
        resized = cv2.resize(face_img, (100,100))
        normalized = resized/255
        reshaped = np.reshape(normalized, (1,100,100,1))
        result = model.predict(reshaped)

        label = np.argmax(result, axis=1)[0]

        cv2.rectangle(img, (x,y), (x+w, y+h), color_dict[label], 3)
        cv2.rectangle(img, (x,y-40), (x+w, y), color_dict[label],-1)
        cv2.putText(img, labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        if label == 0:
            counter += 1
            if counter == trigger:
                requests.get("http://192.168.1.223:3000/shock")
                print('REAL ANGRY')
        else:
            counter = 0
            


    cv2.imshow('Video', img)
    key = cv2.waitKey(1)

    if(key == 27):
        break

cv2.destroyAllWindows()
source.release()