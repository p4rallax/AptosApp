# frontend/main.py

import requests
import streamlit as st
from PIL import Image



# https://discuss.streamlit.io/t/version-0-64-0-deprecation-warning-for-st-file-uploader-decoding/4465
st.set_option("deprecation.showfileUploaderEncoding", False)

# defines an h1 header
st.title("Diabetic Retinopath prediction web app")

# displays a file uploader widget
image = st.file_uploader("Choose an image")

# displays the select widget for the styles
#style = st.selectbox("Choose the style", [i for i in STYLES.keys()])

# displays a button
if st.button("Predict on an image"):
    if image is not None :
        files = {"file": image.getvalue()}
        res = requests.post(f"http://0.0.0.0:8080/pred", files=files)
        path = res.json()
        label = path.get('value')
        #final = path.get(""))
        #st.image(image, width=500)
        st.image(image)
        st.text(label)
