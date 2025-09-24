import streamlit as st
import requests
from PIL import Image

API_URL = "http://localhost:8000/predict"   

def call_api(file_bytes):
    """Gửi ảnh sang API và trả về kết quả"""
    files = {"file": file_bytes}
    try:
        response = requests.post(API_URL, files=files)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API trả về lỗi: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Lỗi kết nối API: {e}")
        return None

def ui_app():
    """Streamlit UI"""
    st.set_page_config(page_title="Vietnamese Food Classifier", page_icon="🍜")
    st.title("🍜 Vietnamese Food Classifier Demo")

    uploaded_file = st.file_uploader("Chọn ảnh món ăn", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh đã chọn", use_container_width=True)

        if st.button("🚀 Dự đoán"):
            with st.spinner("Đang phân loại..."):
                result = call_api(uploaded_file.getvalue())
                if result:
                    st.success(f"Kết quả dự đoán, món ăn này là: {result['predicted_label']} - độ tin cậy ({result['confidence']:.2%})")

if __name__ == "__main__":
    ui_app()
