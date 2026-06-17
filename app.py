import streamlit as st
from evaluate import predict
from PIL import Image
import numpy as np

st.title("🚗 Car License Plate Detection System")

uploaded_file = st.file_uploader(
    "Upload Car Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Detect License Plate"):

        # convert PIL → numpy
        #image_np = np.array(image)
        results, confidences, texts = predict(image)
        annotated_img = results.plot()

        st.image(annotated_img,caption="Detection Result",use_container_width=True)
        for i, (conf, text) in enumerate(zip(confidences, texts), start=1):
            st.write(f"Plate {i}: {conf}% confidence")

            if conf >= 60:
                st.success(f"Detected License Plate: {text}")