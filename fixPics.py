import os
import cv2
import numpy as np

from keras.utils import np_utils

face_path = 'facedata'
choices = os.listdir(face_path)
labels = [i for i in range(len(choices))]

label_dict = dict(zip(choices, labels))

img_size = 100
data = []
target = []

for choice in choices:
    folder_path = os.path.join(face_path, choice)
    img_names = os.listdir(folder_path)

    for img_name in img_names:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)

        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (img_size, img_size))
            data.append(resized)
            target.append(label_dict[choice])

        except Exception as e:
            print("Error: ", e)

data = np.array(data)/255
data = np.reshape(data, (data.shape[0], img_size, img_size,1))
target = np.array(target)

new_target = np_utils.to_categorical(target)

np.save('data', data)
np.save('target', new_target)