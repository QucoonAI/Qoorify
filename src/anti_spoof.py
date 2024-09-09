from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np


def spoof_predict(image_path, model):

    image = load_img(image_path, target_size=(224, 224))
    image_array = img_to_array(image)
    image_array /= 255.0  

    threshold = 0.5  
    prediction = model.predict(np.expand_dims(image_array, axis=0))[0]
    return prediction
    # if prediction > threshold:
    #     return 0
    # else:
    #     return 1
