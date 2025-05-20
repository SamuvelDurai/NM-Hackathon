import streamlit as st
import cv2
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import io

# Load color data
@st.cache_data
def load_colors():
    return pd.read_csv("colors.csv")

# Match closest color name
def get_color_name(R, G, B, df):
    minimum = float("inf")
    cname = "Unknown"
    for i in range(len(df)):
        d = abs(R - int(df.loc[i, "R"])) + abs(G - int(df.loc[i, "G"])) + abs(B - int(df.loc[i, "B"]))
        if d < minimum:
            minimum = d
            cname = df.loc[i, "color_name"]
    return cname

# Extract dominant colors using KMeans
def extract_palette(image, n_colors=5):
    img = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(img)
    colors = np.round(kmeans.cluster_centers_).astype(int)
    return colors

# Streamlit UI
st.set_page_config(page_title="Color Palette Extractor", layout="wide")
st.title("🎨 Color Palette Extractor from Image Region")
st.write("Upload an image and select a region to extract the dominant colors.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
df_colors = load_colors()

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.subheader("Crop Region (pixels)")
    col1, col2 = st.columns(2)
    with col1:
        x1 = st.number_input("Start X", min_value=0, max_value=img_array.shape[1] - 1, value=0)
        y1 = st.number_input("Start Y", min_value=0, max_value=img_array.shape[0] - 1, value=0)
    with col2:
        x2 = st.number_input("End X", min_value=1, max_value=img_array.shape[1], value=img_array.shape[1])
        y2 = st.number_input("End Y", min_value=1, max_value=img_array.shape[0], value=img_array.shape[0])

    if st.button("Extract Palette"):
        if x2 > x1 and y2 > y1:
            region = img_array[int(y1):int(y2), int(x1):int(x2)]
            dominant_colors = extract_palette(region, n_colors=5)

            st.subheader("🎨 Dominant Colors")
            for i, color in enumerate(dominant_colors):
                R, G, B = color
                color_name = get_color_name(R, G, B, df_colors)
                st.markdown(
                    f"""
                    <div style='display:flex;align-items:center;margin-bottom:10px'>
                        <div style='width:40px;height:40px;background-color:rgb({R},{G},{B});border:1px solid #000;margin-right:10px'></div>
                        <div>
                            <strong>{color_name}</strong><br/>
                            RGB: ({R}, {G}, {B})
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.error("Invalid region. Make sure End X > Start X and End Y > Start Y.")
