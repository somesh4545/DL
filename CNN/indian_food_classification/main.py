import tensorflow as tf
import gradio as gr
from tensorflow.keras.preprocessing import image as Image
import numpy as np

model = tf.keras.models.load_model('model.h5')

labels = ['Biryani', 'Chole-Bhature', 'Jalebi', 'Kofta', 'Naan',
          'Paneer-Tikka', 'Pani-Puri', 'Pav-Bhaji', 'Vadapav', 'dal', 'dosa']


def predict_food(image):
    # image = image.resize((256, 256))
    x = Image.img_to_array(image)
    x = np.expand_dims(x, axis=0)
    # result = model.predict(x)
    # print(result)
    # return result
    prediction = model.predict(x).flatten()
    return {labels[i]: float(prediction[i]) for i in range(11)}


image = gr.inputs.Image(shape=(256, 256))
label = gr.outputs.Label(num_top_classes=3)

title = 'Indian Food Classification Model'
description = "This are some of the categories upload any food image or from the below images  'Biryani', 'Chole-Bhature', 'Jalebi', 'Kofta', 'Naan','Paneer-Tikka', 'Pani-Puri', 'Pav-Bhaji', 'Vadapav', 'dal', 'dosa'"

gr.Interface(
    title=title,
    description=description,
    fn=predict_food,
    inputs=image,
    outputs=label,
    examples=[["pani_puri.png"], ["pav_bhaji.png"],
              ["dal.png"], ["Chole-Bhature.png"]]  # in the same directory or trial folder
).launch()
