import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

def import_and_predict(image_data, model):
    
        size = (75,75)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = image.convert('RGB')
        image = np.asarray(image)
        image = (image.astype(np.float32) / 255.0)

        img_reshape = image[np.newaxis,...]

        prediction = model.predict(img_reshape)
        
        return prediction

model = tf.keras.models.load_model('CVA1210E.hdf5') #loading a trained model

st.write("""
         # Chinese number 3-4-5 Hand Sign Prediction
         """
         )

st.write("This is a simple image classification web app to predict Chinese number 3-4-5 hand sign")

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
#
if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("It is Chinese hand gesture 1")
    elif np.argmax(prediction) == 1:
        st.write("It is Chinese hand gesture 2")
    elif np.argmax(prediction) == 2:
        st.write("It is Chinese hand gesture 3")
    elif np.argmax(prediction) == 3:
        st.write("It is Chinese hand gesture 4")
    elif np.argmax(prediction) == 4:
        st.write("It is Chinese hand gesture 5")
    elif np.argmax(prediction) == 5:
        st.write("It is Chinese hand gesture 6")
    elif np.argmax(prediction) == 6:
        st.write("It is Chinese hand gesture 7")
    elif np.argmax(prediction) == 7:
        st.write("It is Chinese hand gesture 8")
    elif np.argmax(prediction) == 8:
        st.write("It is Chinese hand gesture 9")   
    else:
        st.write("It is Chinese hand gesture 10")
    
    st.text("Probability (1,2,3,4,5,6,7,8,9,10)")
    st.write(prediction)
