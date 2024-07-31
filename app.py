import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Define image fusion function
def fuse_images(img1, img2, encoder, decoder, fun="l2"):
    encoded_img1 = encoder.predict(np.expand_dims(img1, axis=0))
    encoded_img2 = encoder.predict(np.expand_dims(img2, axis=0))
    if fun == "add":
        fused_encoded = (encoded_img1 + encoded_img2)
    elif fun == "avg":
        fused_encoded = (encoded_img1 + encoded_img2) / 2
    elif fun == "l1":
        fused_encoded = np.maximum(np.abs(encoded_img1), np.abs(encoded_img2))
    elif fun == "l2":
        fused_encoded = np.sqrt(np.square(encoded_img1) + np.square(encoded_img2))
    fused_image = decoder.predict(fused_encoded)
    return fused_image.squeeze()


# # Load the pre-trained autoencoder model
autoencoder = load_model("autoencoder.h5")
encoder = autoencoder.get_layer('encoder')
decoder = autoencoder.get_layer('decoder')

# Streamlit app
st.title("Image Fusion using Autoencoder")

st.sidebar.header("Upload Images")
uploaded_file1 = st.sidebar.file_uploader("Choose the visible image...", type=["jpg", "jpeg", "png" , "bmp" , "tif"])
uploaded_file2 = st.sidebar.file_uploader("Choose the infrared image...", type=["jpg", "jpeg", "png" , "bmp" , "tif"])
fusion_function = st.sidebar.selectbox("Select Fusion Function", ["add", "avg", "l1", "l2"])

if uploaded_file1 is not None and uploaded_file2 is not None:
    img1 = Image.open(uploaded_file1).convert("L")
    img2 = Image.open(uploaded_file2).convert("L")
    img1 = np.array(img1.resize((256, 256))) / 255.0
    img2 = np.array(img2.resize((256, 256))) / 255.0
    img1 = np.expand_dims(img1, axis=-1)
    img2 = np.expand_dims(img2, axis=-1)
    st.sidebar.image([uploaded_file1, uploaded_file2], caption=["Image 1", "Image 2"], width=128)
    fused_image = fuse_images(img1, img2, encoder, decoder, fusion_function)
    st.subheader("Image 1")
    st.image(img1,width=256,clamp=True,channels="GRAY")
    st.subheader("Image 2")
    st.image(img2,width=256,clamp=True,channels="GRAY")
    st.subheader("Fused Image")
    st.image(fused_image, width=256, clamp=True, channels="GRAY")

if st.sidebar.button("Fuse and Show Images"):
    if uploaded_file1 is not None and uploaded_file2 is not None:
        fused_image = fuse_images(img1, img2, encoder, decoder, fusion_function)
        st.subheader("Image 1")
        st.image(img1,width=256,clamp=True,channels="GRAY")
        st.subheader("Image 2")
        st.image(img2,width=256,clamp=True,channels="GRAY")
        st.subheader("Fused Image")
        st.image(fused_image, width=256, clamp=True, channels="GRAY")
    else:
        st.warning("Please upload two images for fusion.")