import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
from tensorflow.keras.models import load_model

MODELSPATH = './model/'
DATAPATH = './data/'

# Function to render a styled header with a smaller box
def render_header():
    st.markdown("""
        <div style="text-align:center;padding:0.25em;background-color:#5A2F23;">
            <h1 style="color:white;font-size:1.5em; font-weight:bold; text-shadow: 2px 2px 4px #186a3b;">
                <i class="fas fa-medkit" style="color:#f39c12;"></i> Skin Cancer Classifier
            </h1>
            <p style="color:#ecf0f1;font-size:1.2em;">Detecting skin cancer with AI</p>
        </div>
    """, unsafe_allow_html=True)
# Function to load sample data image
@st.cache_data
def load_mekd():
    img = Image.open(DATAPATH + '/Melanocytic.jpg')
    return img

# Function to preprocess image data
@st.cache_data
def data_gen(x):
    img = np.asarray(Image.open(x).resize((28, 28)))
    x_test = (img - np.mean(img)) / np.std(img)
    x_validate = x_test.reshape(1, 28, 28, 3)
    return x_validate

# Function to load Keras model
def load_models():
    model = load_model(MODELSPATH + 'Classifier.h5')
    return model

# Function to predict and display results
@st.cache
def predict(x_test, model):
    Y_pred = model.predict(x_test)
    ynew = np.round(Y_pred, 2) * 100
    y_new = ynew[0].tolist()
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    return y_new, Y_pred_classes

# Function to display prediction results in a styled way
@st.cache
def display_prediction(y_new):
    result = pd.DataFrame({'Probability': y_new}, index=np.arange(7))
    result = result.reset_index()
    result.columns = ['Classes', 'Probability']
    lesion_type_dict = {2: 'Benign keratosis-like lesions', 4:'Melanoma', 3: 'Dermatofibroma',
                        5: 'Melanocytic nevi' , 6: 'Vascular lesions', 1: 'Basal cell carcinoma', 0: 'Actinic keratoses'}
    result["Classes"] = result["Classes"].map(lesion_type_dict)
    return result

# Main function
def main():
    st.set_page_config(
        page_title="Skin Cancer Classifier",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    render_header()

    st.sidebar.header('Navigation')
    page = st.sidebar.radio("Go to", ["Sample Data", "Upload Your Image"])

    if page == "Sample Data":
        st.header("Sample Data Prediction for Skin Cancer")
        st.markdown("""
        **Get Predictions using Sample Data**
        Choose a Sample Data image and see the predictions.
        """)
        mov_base = ['Sample Data I']
        movies_chosen = st.multiselect('Choose Sample Data', mov_base)

        if len(movies_chosen) == 1:
            st.success("You have selected Sample Data")
            if st.checkbox('Show Sample Data'):
                st.info("Showing Sample data ---->>>")
                image = load_mekd()
                st.image(image, caption='Sample Data', use_column_width=True)
                st.subheader("Choose Training Algorithm!")
                if st.checkbox('Keras'):
                    model = load_models()
                    st.success("Keras Model Loaded!")
                    if st.checkbox('Show Prediction Probability on Sample Data'):
                        x_test = data_gen(DATAPATH + '/Melanocytic.jpg')
                        y_new, Y_pred_classes = predict(x_test, model)
                        result = display_prediction(y_new)
                        st.write(result)
                        if st.checkbox('Display Probability Graph'):
                            fig = px.bar(result, x="Classes",
                                         y="Probability", color='Classes')
                            st.plotly_chart(fig, use_container_width=True)

    if page == "Upload Your Image":
        st.header("Upload Your Image")

        file_path = st.file_uploader('Upload an image', type=['png', 'jpg','jpeg'])

        if file_path is not None:
            x_test = data_gen(file_path)
            image = Image.open(file_path)
            img_array = np.array(image)

            st.success('File Upload Success!!')
        else:
            st.info('Please upload an Image file')

        if st.checkbox('Show Uploaded Image'):
            st.info("Showing Uploaded Image ---->>>")
            st.image(img_array, caption='Uploaded Image',
                     use_column_width=True)
            st.subheader("Choose Training Algorithm!")
            if st.checkbox('Keras'):
                model = load_models()
                st.success("Keras Model Loaded!")
                if st.checkbox('Show Prediction Probability for Uploaded Image'):
                    y_new, Y_pred_classes = predict(x_test, model)
                    result = display_prediction(y_new)
                    st.write(result)
                    if st.checkbox('Display Probability Graph'):
                        fig = px.bar(result, x="Classes",
                                     y="Probability", color='Classes')
                        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
