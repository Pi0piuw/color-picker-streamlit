# Nama : Shofy Aliya
# NPM : 140810230063
# Deskripsi : Website Color Picker

## Import Library
import streamlit as st
import numpy as np
import cv2
from sklearn.cluster import KMeans
from PIL import Image
from io import BytesIO
import matplotlib.colors as mcolors

## Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Color Picker dari Gambar", layout="centered")

st.title("🎨 Color Picker dari Gambar")
st.write("Unggah gambar dan dapatkan 5 warna dominan dalam bentuk palet!")

## Upload Gambar
uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

## Fungsi untuk Mengambil Warna Dominan
def get_palette(image, n_colors=5):
    img = np.array(image)
    img = cv2.resize(img, (100, 100))  # Perkecil untuk kecepatan
    img = img.reshape((img.shape[0] * img.shape[1], 3))

    kmeans = KMeans(n_clusters=n_colors)
    kmeans.fit(img)

    colors = kmeans.cluster_centers_.astype(int)
    return colors

## Fungsi untuk Konversi Warna
def rgb_to_hex(rgb_color):
    return mcolors.to_hex([c / 255 for c in rgb_color])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang Diupload", use_container_width=True)

    with st.spinner("Menganalisis warna dominan..."):
        colors = get_palette(image)

    st.subheader("🎯 Warna Dominan:")
    cols = st.columns(len(colors))
    for i, col in enumerate(cols):
        rgb = colors[i]
        hex_code = rgb_to_hex(rgb)
        with col:
            st.color_picker(f"Warna #{i+1}", hex_code, key=f"picker_{i}")
            st.write(hex_code.upper())
