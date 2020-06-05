import numpy as np
from tensorflow.keras.models import load_model


class FacialExpressionModel(object):
    def __init__(self,model_file):
        self.emotions_list=['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
        self.loaded_model=load_model(model_file)

    def predict_emotion(self,img):
        self.preds=self.loaded_model.predict(img)
        return self.emotions_list[np.argmax(self.preds)]