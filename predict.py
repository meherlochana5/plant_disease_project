import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model("plant_model.h5")

# class names (same order as your training generator printed)
class_names = [
    'Potato___Early_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

# test image path
img_path = 'health.jpg'   # change picture name if needed

# preprocessing
img = image.load_img(img_path, target_size=(150,150))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# prediction
pred = model.predict(img_array)
index = np.argmax(pred)

print("Predicted Disease:", class_names[index])
