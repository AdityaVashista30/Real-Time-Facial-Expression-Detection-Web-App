# Real-Time-Facial-Expression-Detection-Web-App
Real time sentiment analysis of face using Keras and OpenCv with help of web application based on Flask
The web app is able to detect the following 7 emotions in reeal time:

    1.Happy
    2.Angry
    3.Disgust
    4.Fear
    5.Sad
    6.Surrprise
    7.Neutral
 
The detection model used a combination of CNN layers followed by ANN layers, having tesnorflow.keras backend. 
Face detection is done via  CV2 library.
The template for web app is stored in tempelates folder as index.html
We can also detect emotions of people in a recorded video by changing only a single line (line 11) in camera.py. (The line has been commited out in cmera.py itself for video usage)

Execution flow: Facial_Expression_Training.py (to build and store the model in .h5 format) 
                ->model.py
                ->camera.py
                ->main.py
