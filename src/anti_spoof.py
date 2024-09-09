from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

model = load_model("./model/face_antispoofing_model.h5")

def spoof_prdict(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image_array = img_to_array(image)
    image_array /= 255.0  

    threshold = 0.5  
    prediction = model.predict(np.expand_dims(image_array, axis=0))[0]
    if prediction > threshold:
        return "real"
    else:
        return "spoof"
