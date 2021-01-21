# Airsoft-Robot
A robot that shoots you when you get mad in a video game

This is the software that does the facial detection and determines if you are angry or not using machine learning
the fixPics.py takes all the pictures in a folder and turning them to gray scale and makes the picture smaller then uses numpy to save the data into a .npy it will save a data.npy and a target.npy
/br
the emotionModel.py takes data.npy and target.npy and uses it to make a machine learning model using TensorFlow and Keras and saves the models
/br
the emotinoDetector.py uses OpenCV to get access to my webcam and i use the haarcascade to detect the face if its there and i use the model to determine if the user is making a happy, neutral or angry face.
/br
i have a video on youtube demostrating how the robot works.

youtube video : https://www.youtube.com/watch?v=J64MtaqgVCI&t=18s
