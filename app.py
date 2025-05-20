import streamlit as st
import pandas as pd
import cv2
import numpy as np
from streamlit_image_coordinates import streamlit_image_coordinates

# Load color dataset
@st.cache_data
def load_colors():
    return pd.read_csv("colors.csv")

df = load_colors()

def get_color_name(R, G, B):
    minimum = float('inf')
    cname = ""
    for i in range(len(df)):
        d = abs(R - int(df.loc[i, "R"])) + abs(G - int(df.loc[i, "G"])) + abs(B - int(df.loc[i, "B"]))
        if d < minimum:
            minimum = d
            cname = df.loc[i, "color_name"]
    return cname

# UI
st.title("Hover-Based Color Detection App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize for UI
    img_resized = cv2.resize(img, (600, 400))

    # Show image and get coordinates
    coords = streamlit_image_coordinates(img_resized, key="hover")

    if coords is not None:
        x, y = int(coords["x"]), int(coords["y"])
        if 0 <= x < img_resized.shape[1] and 0 <= y < img_resized.shape[0]:
            R, G, B = img_resized[y, x]
            color_name = get_color_name(R, G, B)

            st.markdown(f"**Hovered Pixel at ({x}, {y})**")
            st.markdown(f"**Color Name**: `{color_name}`")
            st.markdown(f"**RGB**: ({R}, {G}, {B})")
            st.markdown(
                f'<div style="width:100px;height:50px;background-color:rgb({R},{G},{B});border:1px solid #000;"></div>',
                unsafe_allow_html=True
            )
