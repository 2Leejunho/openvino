import streamlit as st
import PIL
import cv2
import numpy
import rrr
 
st.set_page_config(
    page_title="Age/Gender/Emotion",
    page_icon="sun_with_face",
    layout="centered",
    initial_sidebar_state="expanded",)

st.title("Age/Gender/Emotion Project :sun_with_face")

st.header("Type")
source_radio = st.sidebar.radio("Select Source",["MIAGE","VIDEO","WEBCAM"])

inputt = None
if source_radio == "MIAGE":
    st.sidebar.header("Upload")
    input = st.sidebar.file_uploader("Choose an image.",type =("jpg","png"))

    if input is not None:
        uploaded_image = PIL.Image.open(input)
        uploaded_image_cv = cv2.cvtColor(numpy.array(uploaded_image),cv2.COLOR_RGB2BGR)
        visualized_image = rrr.predict_image(uploaded_image_cv, conf_threshold = 2)
        
        st.image(uploaded_image, channals = "BGR")
