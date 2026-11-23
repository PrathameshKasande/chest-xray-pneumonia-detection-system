
from pathlib import Path

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd

from PIL import Image


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Chest X-ray Pneumonia Detection System | AI Dashboard",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. PROJECT PATHS AND SETTINGS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_custom_cnn.keras"

DATASET_DIR = BASE_DIR / "data" / "raw" / "chest_xray"

IMAGE_SIZE = (224, 224)

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


# =========================================================
# 3. CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #0b1120;
    color: #f1f5f9;
}

[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #263247;
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #f8fafc;
    margin-bottom: 0px;
}

.subtitle {
    color: #94a3b8;
    font-size: 16px;
    margin-top: 5px;
    margin-bottom: 25px;
}

.section-title {
    font-size: 23px;
    font-weight: 700;
    color: #f8fafc;
    margin-top: 15px;
}

.small-muted {
    color: #94a3b8;
    font-size: 13px;
}

div[data-testid="stMetric"] {
    background: #172033;
    border: 1px solid #29364d;
    padding: 18px;
    border-radius: 12px;
}

div[data-testid="stFileUploader"] {
    background-color: #172033;
    border: 1px dashed #526580;
    border-radius: 12px;
    padding: 12px;
}

.stButton > button {
    background: linear-gradient(90deg, #2563eb, #0ea5e9);
    color: white;
    border: none;
    border-radius: 9px;
    padding: 12px 20px;
    font-weight: 700;
    width: 100%;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #1d4ed8, #0284c7);
    color: white;
    border: none;
}

hr {
    border-color: #29364d;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 4. LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(MODEL_PATH)


# =========================================================
# 5. IMAGE PREDICTION FUNCTION
# =========================================================

def predict_xray(uploaded_image):

    img = uploaded_image.convert("RGB")

    img = img.resize(IMAGE_SIZE)

    img_array = np.array(
        img,
        dtype=np.float32
    )

    # Normalize image pixels to 0-1.
    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    prediction = model.predict(
        img_array,
        verbose=0
    )

    prediction = np.asarray(prediction)

    if prediction.size != 1:
        raise ValueError(
            "This application expects a binary model "
            "with one sigmoid output."
        )

    pneumonia_score = float(prediction.reshape(-1)[0])

    if not 0 <= pneumonia_score <= 1:
        raise ValueError(
            "Model output is outside the expected sigmoid range."
        )

    normal_score = 1.0 - pneumonia_score

    if pneumonia_score >= 0.5:
        predicted_class = "PNEUMONIA"
        model_score = pneumonia_score
    else:
        predicted_class = "NORMAL"
        model_score = normal_score

    return {
        "predicted_class": predicted_class,
        "pneumonia_score": pneumonia_score,
        "normal_score": normal_score,
        "model_score": model_score
    }


# =========================================================
# 6. DATASET STATISTICS
# =========================================================

def get_dataset_statistics():

    splits = ["train", "val", "test"]

    statistics = []

    for split in splits:

        split_dir = DATASET_DIR / split

        normal_count = 0
        pneumonia_count = 0

        for class_name in CLASS_NAMES:

            class_dir = split_dir / class_name

            if class_dir.exists():

                image_count = sum(
                    1 for p in class_dir.iterdir()
                    if p.is_file()
                    and p.suffix.lower() in (
                        ".jpg", ".jpeg", ".png"
                    )
                )

                if class_name == "NORMAL":
                    normal_count = image_count
                else:
                    pneumonia_count = image_count

        statistics.append({
            "Dataset split": split.upper(),
            "NORMAL": normal_count,
            "PNEUMONIA": pneumonia_count
        })

    return pd.DataFrame(statistics)


# =========================================================
# 7. SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("# 🫁 Chest X-ray")

    st.markdown("### Pneumonia Detection System")

    st.caption("AI-powered Chest X-ray Analysis")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Dataset Insights",
            "About Project"
        ]
    )

    st.divider()

    st.markdown("### Model Information")

    st.markdown("""
    **Architecture:** CNN

    **Framework:** TensorFlow / Keras

    **Input size:** 224 × 224

    **Task:** Binary Classification

    **Classes:**
    - NORMAL
    - PNEUMONIA
    """)

    st.divider()

    st.caption("Chest X-ray Pneumonia Detection System")


# =========================================================
# 8. LOAD MODEL
# =========================================================

try:

    model = load_model()

except Exception as error:

    st.error(
        f"Unable to load the trained model: {error}"
    )

    st.info(
        "Check that best_custom_cnn.keras exists inside the models folder."
    )

    st.stop()


# =========================================================
# 9. DASHBOARD PAGE
# =========================================================

if page == "Dashboard":

    st.markdown(
        '<div class="main-title">🫁 Chest X-ray Pneumonia Detection System</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Intelligent Chest X-ray Classification using Deep Learning'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    # Top metric cards.
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "AI Architecture",
        "CNN",
        "Deep Learning"
    )

    col2.metric(
        "Input Resolution",
        "224 × 224",
        "RGB Image"
    )

    col3.metric(
        "Classification",
        "2 Classes",
        "Binary Prediction"
    )

    st.markdown(
        '<div class="section-title">Upload & Analyze</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload a chest X-ray image to generate a prediction "
        "using the trained convolutional neural network."
    )

    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG and PNG"
    )

    if uploaded_file is not None:

        try:

            uploaded_image = Image.open(
                uploaded_file
            ).convert("RGB")

            st.markdown("### Image Preview")

            image_col, info_col = st.columns(
                [1.3, 1]
            )

            with image_col:

                st.image(
                    uploaded_image,
                    caption="Uploaded Chest X-ray",
                    use_container_width=True
                )

            with info_col:

                st.markdown("#### Image Information")

                st.write(
                    f"**Filename:** {uploaded_file.name}"
                )

                st.write(
                    f"**Dimensions:** "
                    f"{uploaded_image.width} × "
                    f"{uploaded_image.height}"
                )

                st.write(
                    f"**Format:** {uploaded_file.type}"
                )

                st.write(
                    "**Model input:** 224 × 224 RGB"
                )

            st.divider()

            if st.button(
                "🔍 Analyze Chest X-ray",
                use_container_width=True
            ):

                with st.spinner(
                    "CNN model is analyzing the image..."
                ):

                    result = predict_xray(
                        uploaded_image
                    )

                    st.session_state["prediction_result"] = result
                    st.session_state["prediction_filename"] = uploaded_file.name

        except Exception as error:

            st.error(
                f"Image processing error: {error}"
            )

    # Display saved prediction.
    if (
        "prediction_result" in st.session_state
        and uploaded_file is not None
        and st.session_state.get("prediction_filename") == uploaded_file.name
    ):

        result = st.session_state["prediction_result"]

        st.divider()

        st.markdown(
            '<div class="section-title">Prediction Analysis</div>',
            unsafe_allow_html=True
        )

        result_col, score_col = st.columns([1, 1])

        with result_col:

            if result["predicted_class"] == "PNEUMONIA":

                st.error(
                    "⚠️ Model Prediction: PNEUMONIA"
                )

            else:

                st.success(
                    "✓ Model Prediction: NORMAL"
                )

            st.metric(
                "Predicted Class",
                result["predicted_class"]
            )

        with score_col:

            st.metric(
                "Selected Class Score",
                f"{result['model_score'] * 100:.2f}%"
            )

            st.caption(
                "This is the model's output score, "
                "not a calibrated medical confidence."
            )

        # Probability chart.
        st.markdown("### Class Probability Comparison")

        probability_df = pd.DataFrame({
            "Class": ["NORMAL", "PNEUMONIA"],
            "Model score": [
                result["normal_score"],
                result["pneumonia_score"]
            ]
        })

        st.bar_chart(
            probability_df.set_index("Class"),
            y_label="Model score",
            x_label="Class"
        )

        st.markdown("#### Prediction Scores")

        score1, score2 = st.columns(2)

        score1.metric(
            "NORMAL score",
            f"{result['normal_score'] * 100:.2f}%"
        )

        score2.metric(
            "PNEUMONIA score",
            f"{result['pneumonia_score'] * 100:.2f}%"
        )

        st.info(
            "The chart displays the model's two class scores. "
            "These scores should not be interpreted as a medical diagnosis."
        )

    st.divider()

    st.markdown("### How the System Works")

    step1, step2, step3 = st.columns(3)

    with step1:

        st.markdown("#### 01 · Upload")

        st.write(
            "Upload a chest X-ray image in JPG, JPEG or PNG format."
        )

    with step2:

        st.markdown("#### 02 · Process")

        st.write(
            "The image is resized and normalized before being passed to the CNN."
        )

    with step3:

        st.markdown("#### 03 · Predict")

        st.write(
            "The trained model returns scores for NORMAL and PNEUMONIA."
        )


# =========================================================
# 10. DATASET INSIGHTS PAGE
# =========================================================

elif page == "Dataset Insights":

    st.markdown(
        '<div class="main-title">📊 Dataset Insights</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Explore the image distribution used in the project'
        '</div>',
        unsafe_allow_html=True
    )

    if not DATASET_DIR.exists():

        st.warning(
            "Dataset directory was not found. "
            "Check data/raw/chest_xray."
        )

    else:

        df = get_dataset_statistics()

        total_normal = int(df["NORMAL"].sum())

        total_pneumonia = int(df["PNEUMONIA"].sum())

        total_images = total_normal + total_pneumonia

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Images",
            f"{total_images:,}"
        )

        col2.metric(
            "NORMAL Images",
            f"{total_normal:,}"
        )

        col3.metric(
            "PNEUMONIA Images",
            f"{total_pneumonia:,}"
        )

        st.divider()

        st.markdown("### Dataset Distribution by Split")

        chart_df = df.set_index("Dataset split")

        st.bar_chart(
            chart_df,
            y_label="Number of images",
            x_label="Dataset split"
        )

        st.markdown("### Dataset Summary")

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### Class Distribution")

        class_df = pd.DataFrame({
            "Class": ["NORMAL", "PNEUMONIA"],
            "Number of Images": [
                total_normal,
                total_pneumonia
            ]
        })

        st.bar_chart(
            class_df.set_index("Class"),
            y_label="Number of images"
        )

        st.caption(
            "Counts are calculated from the image files "
            "currently present in your dataset folders."
        )


# =========================================================
# 11. ABOUT PROJECT PAGE
# =========================================================

elif page == "About Project":

    st.markdown(
        '<div class="main-title">💡 About the Project</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Project overview, technology stack and workflow'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "### Chest X-ray Pneumonia Detection System"
    )

    st.write("""
    The Chest X-ray Pneumonia Detection System is a deep learning
    project that uses a Convolutional Neural Network (CNN) to
    classify chest X-ray images into two categories:
    NORMAL and PNEUMONIA.

    The project demonstrates image preprocessing,
    CNN model training, model evaluation and image prediction
    through an interactive Streamlit interface.
    """)

    st.divider()

    st.markdown("### Technology Stack")

    tech1, tech2, tech3 = st.columns(3)

    tech1.markdown("#### Python")
    tech1.write("Programming language")

    tech2.markdown("#### TensorFlow")
    tech2.write("Deep learning framework")

    tech3.markdown("#### Streamlit")
    tech3.write("Interactive web interface")

    tech4, tech5, tech6 = st.columns(3)

    tech4.markdown("#### NumPy")
    tech4.write("Numerical operations")

    tech5.markdown("#### Matplotlib")
    tech5.write("Data visualization")

    tech6.markdown("#### Scikit-learn")
    tech6.write("Model evaluation")

    st.divider()

    st.markdown("### Project Workflow")

    st.markdown("""
    1. **Data Exploration** – Explore the chest X-ray dataset.
    2. **Preprocessing** – Resize and normalize images.
    3. **CNN Training** – Train a convolutional neural network.
    4. **Evaluation** – Evaluate model performance.
    5. **Grad-CAM** – Visualize image regions contributing to model predictions.
    6. **Prediction** – Predict classes for uploaded X-ray images.
    7. **Deployment Interface** – Present predictions through Streamlit.
    """)

    st.divider()

    st.markdown("### Future Improvements")

    st.markdown("""
    - Add Grad-CAM visualization directly to the web interface.
    - Add confusion matrix and ROC curve from saved evaluation results.
    - Improve model generalization and external validation.
    - Deploy the application online for demonstration.
    """)


# =========================================================
# 12. FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#94a3b8; padding:15px;">
        <b>Chest X-ray Pneumonia Detection System</b><br>
        CNN-Based Chest X-ray Classification Project<br>
        <span style="font-size:12px;">
        Built with Python · TensorFlow · Streamlit
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

st.warning(
    "Educational and research demonstration only. "
    "This system is not a medical device. Its predictions "
    "must not be used for diagnosis or treatment decisions."
)
