import streamlit as st

st.set_page_config(
    page_title="Etsy AI",
    page_icon="💎",
    layout="wide"
)

st.title("💎 Etsy AI")
st.subheader("AI Jewelry Listing & Image Studio")

st.write(
    "Upload your jewelry images and generate product analysis, "
    "Etsy SEO copy, and marketing content."
)

uploaded_files = st.file_uploader(
    "Upload Product Images",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

if uploaded_files:
    st.success(f"{len(uploaded_files)} image(s) uploaded.")
