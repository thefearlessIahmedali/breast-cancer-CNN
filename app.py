import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    Input,
    Multiply,
    GlobalAveragePooling2D,
    Reshape,
    UpSampling2D
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="OncoVision AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# MODERN CSS DESIGN
# ============================================================

st.markdown("""

<style>

/* Global */

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}


.stApp {

    background:
    radial-gradient(
        circle at top left,
        #1e3a8a 0%,
        transparent 35%
    ),
    radial-gradient(
        circle at bottom right,
        #7c3aed 0%,
        transparent 35%
    ),
    #020617;

    color:white;
}



/* Hide Streamlit default */

header {
    visibility:hidden;
}


footer {
    visibility:hidden;
}


/* Main container */

.main {

    padding-top:2rem;

}



/* Hero */

.hero {


    background:
    linear-gradient(
        135deg,
        rgba(255,255,255,0.12),
        rgba(255,255,255,0.04)
    );


    border-radius:25px;

    padding:45px;

    margin-bottom:35px;

    backdrop-filter:blur(20px);

    border:
    1px solid rgba(255,255,255,0.15);

}


.hero h1 {


    font-size:55px;

    font-weight:800;

    margin-bottom:10px;

}


.gradient {

    background:
    linear-gradient(
        90deg,
        #38bdf8,
        #a78bfa
    );

    -webkit-background-clip:text;

    color:transparent;

}


.hero p {

    color:#cbd5e1;

    font-size:18px;

}



/* Cards */


.card {


    background:

    rgba(255,255,255,0.08);


    border-radius:25px;


    padding:30px;


    border:

    1px solid rgba(255,255,255,0.15);


    backdrop-filter:

    blur(20px);


    box-shadow:

    0 20px 50px rgba(0,0,0,.25);


}



/* Card Titles */

.card h2 {

    font-size:28px;

    margin-bottom:20px;

}



.label {


    color:#38bdf8;

    font-size:13px;

    font-weight:700;

    letter-spacing:2px;

}




/* Upload */


[data-testid="stFileUploader"] {


    background:

    rgba(255,255,255,0.05);


    border-radius:20px;


    padding:15px;

}




/* Buttons */


.stButton button {


    width:100%;


    background:

    linear-gradient(
        90deg,
        #06b6d4,
        #8b5cf6
    );


    color:white;


    border:none;


    padding:15px;


    border-radius:15px;


    font-size:18px;


    font-weight:700;


    transition:.3s;

}



.stButton button:hover {


    transform:translateY(-3px);


    box-shadow:

    0 15px 30px rgba(139,92,246,.5);


}




/* Result badges */


.success-box {


    padding:20px;

    border-radius:20px;

    background:

    rgba(34,197,94,.15);

    border:

    1px solid #22c55e;

}


.warning-box {


    padding:20px;

    border-radius:20px;

    background:

    rgba(239,68,68,.15);

    border:

    1px solid #ef4444;

}


.normal-box {


    padding:20px;

    border-radius:20px;

    background:

    rgba(56,189,248,.15);

    border:

    1px solid #38bdf8;

}



/* Sidebar */


section[data-testid="stSidebar"] {


    background:

    linear-gradient(
        180deg,
        #020617,
        #111827
    );


}



.sidebar-title {


    font-size:25px;

    font-weight:800;


}



</style>

""", unsafe_allow_html=True)



# ============================================================
# SIDEBAR
# ============================================================


with st.sidebar:


    st.markdown(
        """
        <div class="sidebar-title">
        🧬 OncoVision AI
        </div>
        """,
        unsafe_allow_html=True
    )


    st.write("")


    st.info(
        """
        **Deep Learning Breast Cancer Detection**

        Model:
        Attention-CNN

        Input:
        Ultrasound Image

        Classes:
        • Benign
        • Malignant
        • Normal
        """
    )


    st.warning(
        """
        ⚠️ Research Prototype

        This system is not a medical diagnosis tool.
        Always consult qualified healthcare professionals.
        """
    )



# ============================================================
# MODEL ARCHITECTURE
# ============================================================


def attention_module(x):

    avg_pool = GlobalAveragePooling2D()(x)

    avg_pool = Reshape(
        (1,1,avg_pool.shape[1])
    )(avg_pool)


    attention = Conv2D(
        filters=x.shape[-1],
        kernel_size=(1,1),
        activation="sigmoid",
        padding="same"
    )(avg_pool)



    attention = UpSampling2D(
        size=(x.shape[1],x.shape[2]),
        interpolation="bilinear"
    )(attention)


    return Multiply()([x,attention])



def create_cnn_model(input_shape,num_classes):


    inputs = Input(shape=input_shape)


    x = Conv2D(
        32,
        (3,3),
        activation="relu",
        padding="same"
    )(inputs)


    x = MaxPooling2D()(x)


    x = attention_module(x)



    x = Conv2D(
        64,
        (3,3),
        activation="relu",
        padding="same"
    )(x)


    x = MaxPooling2D()(x)


    x = attention_module(x)



    x = Conv2D(
        128,
        (3,3),
        activation="relu",
        padding="same"
    )(x)


    x = MaxPooling2D()(x)


    x = attention_module(x)



    x = Flatten()(x)


    x = Dense(
        256,
        activation="relu"
    )(x)


    x = Dropout(.5)(x)



    outputs = Dense(
        num_classes,
        activation="softmax"
    )(x)



    return Model(
        inputs,
        outputs
    )



@st.cache_resource
def load_my_model():

    model=create_cnn_model(
        (224,224,3),
        3
    )


    model.load_weights(
        "breast_cancer_weights.weights.h5"
    )


    return model



with st.spinner("🧠 Loading Attention-CNN Model..."):

    model=load_my_model()



class_labels = {

    0:"Benign",

    1:"Malignant",

    2:"Normal"

}



# PAGE STATE

if "page" not in st.session_state:
    st.session_state.page="upload"


if "predictions" not in st.session_state:
    st.session_state.predictions=None


if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image=None
    # ============================================================
# UPLOAD PAGE
# ============================================================


def show_upload_page():


    st.markdown(
        """
        <div class="hero">

        <h1>
        🧬 Breast Cancer 
        <span class="gradient">
        Detection AI
        </span>
        </h1>


        <p>
        Advanced Attention-CNN deep learning system
        for breast ultrasound image classification.
        Upload a scan and let AI analyze tissue patterns.
        </p>


        </div>
        """,
        unsafe_allow_html=True
    )



    col1,col2 = st.columns(
        [1,1],
        gap="large"
    )



    # ==========================
    # UPLOAD CARD
    # ==========================


    with col1:


        st.markdown(
            """
            <div class="card">

            <div class="label">
            STEP 01
            </div>

            <h2>
            📤 Upload Ultrasound Scan
            </h2>

            """,
            unsafe_allow_html=True
        )


        uploaded_file = st.file_uploader(
            "Choose ultrasound image",
            type=[
                "jpg",
                "jpeg",
                "png"
            ]
        )


        if uploaded_file:


            image = Image.open(
                uploaded_file
            ).convert("RGB")


            st.session_state.uploaded_image=image



            st.image(
                image,
                caption="Uploaded Scan",
                use_container_width=True
            )



        else:

            st.info(
                "Waiting for ultrasound image..."
            )



        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )




    # ==========================
    # ANALYSIS CARD
    # ==========================


    with col2:


        st.markdown(
            """
            <div class="card">


            <div class="label">
            STEP 02
            </div>


            <h2>
            🤖 AI Analysis
            </h2>

            """,
            unsafe_allow_html=True
        )



        if st.session_state.uploaded_image is None:


            st.warning(
                "Please upload an image first."
            )


        else:


            st.success(
                """
                Image ready.

                Attention-CNN model is prepared
                for analysis.
                """
            )


            if st.button(
                "🚀 Run AI Detection"
            ):


                with st.spinner(
                    "Analyzing tissue features..."
                ):


                    img = (
                        st.session_state
                        .uploaded_image
                        .resize(
                            (224,224)
                        )
                    )


                    img_array=np.array(img)/255.0


                    img_array=np.expand_dims(
                        img_array,
                        axis=0
                    )


                    predictions=model.predict(
                        img_array
                    )[0]



                st.session_state.predictions=predictions


                st.session_state.page="results"


                st.rerun()



        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )





# ============================================================
# RESULTS PAGE
# ============================================================



def show_results_page():



    st.markdown(
        """
        <div class="hero">


        <h1>
        🔬 AI Analysis
        <span class="gradient">
        Complete
        </span>
        </h1>


        <p>
        Attention-CNN classification results
        from your ultrasound scan.
        </p>


        </div>

        """,
        unsafe_allow_html=True
    )



    if st.button(
        "← Analyze Another Image"
    ):


        st.session_state.page="upload"

        st.session_state.predictions=None

        st.rerun()





    col1,col2 = st.columns(
        [1,1.2],
        gap="large"
    )



    # IMAGE


    with col1:


        st.markdown(
            """
            <div class="card">


            <div class="label">
            SCAN
            </div>


            <h2>
            Ultrasound Preview
            </h2>

            """,
            unsafe_allow_html=True
        )



        st.image(
            st.session_state.uploaded_image,
            use_container_width=True
        )


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )





    # RESULTS


    with col2:


        st.markdown(
            """
            <div class="card">


            <div class="label">
            AI RESULTS
            </div>


            <h2>
            Classification Probability
            </h2>

            """,
            unsafe_allow_html=True
        )



        predictions = (
            st.session_state.predictions
        )



        for idx,label in class_labels.items():


            confidence=float(
                predictions[idx]*100
            )



            st.write(
                f"### {label}"
            )


            st.progress(
                int(confidence)
            )


            st.write(
                f"{confidence:.2f}% confidence"
            )




        st.divider()



        final_index=np.argmax(
            predictions
        )


        result=class_labels[
            final_index
        ]



        st.markdown(
            "## Final AI Decision"
        )



        if result=="Malignant":


            st.markdown(
                """
                <div class="warning-box">

                ⚠️
                <h3>
                Malignant Pattern Detected
                </h3>

                The model detected patterns
                associated with malignant tissue.

                Clinical evaluation is recommended.

                </div>
                """,
                unsafe_allow_html=True
            )



        elif result=="Benign":


            st.markdown(
                """
                <div class="success-box">

                🔎

                <h3>
                Benign Pattern Detected
                </h3>

                The model detected patterns
                commonly associated with benign tissue.

                </div>
                """,
                unsafe_allow_html=True
            )



        else:


            st.markdown(
                """
                <div class="normal-box">

                ✅

                <h3>
                Normal Tissue Pattern
                </h3>


                The model classified the image
                as normal tissue.

                </div>
                """,
                unsafe_allow_html=True
            )



        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )





# ============================================================
# APPLICATION ROUTER
# ============================================================


if st.session_state.page=="upload":

    show_upload_page()


else:

    show_results_page()