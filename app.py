import os
import streamlit as st
# The Keras model (Trained_eye_disease_model.h5) is converted to ONNX so the app
# runs with the lightweight onnxruntime instead of TensorFlow.
# MobileNetV3 has its preprocessing built into the model, so raw 0-255 pixels go in.
import onnxruntime as ort
import numpy as np
from PIL import Image
from recommendation import cnv,dme,drusen,normal

st.set_page_config(page_title="Eye Disease Detection", page_icon="👁️", layout="wide")

CLASS_NAMES = ['CNV','DME','DRUSEN','NORMAL']
CLASS_INFO = {
    'CNV': ("Choroidal Neovascularization", "OCT scan showing *CNV with subretinal fluid.*", cnv),
    'DME': ("Diabetic Macular Edema", "OCT scan showing *DME with retinal thickening and intraretinal fluid.*", dme),
    'DRUSEN': ("Drusen (Early AMD)", "OCT scan showing *drusen deposits in early AMD.*", drusen),
    'NORMAL': ("Normal Retina", "OCT scan showing a *normal retina with preserved foveal contour.*", normal),
}
SAMPLES_DIR = "samples"

@st.cache_resource()
def load_model():
    return ort.InferenceSession("eye_disease_model.onnx")

#Model Prediction - returns the probability for each class
def model_prediction(image):
    model = load_model()
    img = image.convert("RGB").resize((224,224))
    x = np.asarray(img, dtype=np.float32)
    x = np.expand_dims(x,axis=0)
    return model.run(None, {model.get_inputs()[0].name: x})[0][0]

def list_samples():
    samples = {}
    if os.path.isdir(SAMPLES_DIR):
        for cls in CLASS_NAMES:
            folder = os.path.join(SAMPLES_DIR, cls)
            if os.path.isdir(folder):
                for f in sorted(os.listdir(folder)):
                    samples[f"{cls} - {f}"] = os.path.join(folder, f)
    return samples

#UI Part

# Sidebar
st.sidebar.title("👁️ Eye Disease Detection Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Identification"])
st.sidebar.markdown("---")
st.sidebar.caption("Classifies retinal OCT scans into CNV, DME, Drusen or Normal using a MobileNetV3 model.")
st.sidebar.warning("For educational use only. Not a substitute for a medical diagnosis.")


#Main Page
if(app_mode == "Home"):
     st.markdown("""
    ## **OCT Retinal Analysis Platform**

#### **Welcome to the Retinal OCT Analysis Platform**

**Optical Coherence Tomography (OCT)** is a powerful imaging technique that provides high-resolution cross-sectional images of the retina, allowing for early detection and monitoring of various retinal diseases. Each year, over 30 million OCT scans are performed, aiding in the diagnosis and management of eye conditions that can lead to vision loss, such as choroidal neovascularization (CNV), diabetic macular edema (DME), and age-related macular degeneration (AMD).

##### **Why OCT Matters**
OCT is a crucial tool in ophthalmology, offering non-invasive imaging to detect retinal abnormalities. On this platform, we aim to streamline the analysis and interpretation of these scans, reducing the time burden on medical professionals and increasing diagnostic accuracy through advanced automated analysis.

---

#### **Key Features of the Platform**

- **Automated Image Analysis**: Our platform uses state-of-the-art machine learning models to classify OCT images into distinct categories: **Normal**, **CNV**, **DME**, and **Drusen**.
- **Cross-Sectional Retinal Imaging**: Examine high-quality images showcasing both normal retinas and various pathologies, helping doctors make informed clinical decisions.
- **Streamlined Workflow**: Upload, analyze, and review OCT scans in a few easy steps.

---

#### **Understanding Retinal Diseases through OCT**

1. **Choroidal Neovascularization (CNV)**
   - Neovascular membrane with subretinal fluid
   
2. **Diabetic Macular Edema (DME)**
   - Retinal thickening with intraretinal fluid
   
3. **Drusen (Early AMD)**
   - Presence of multiple drusen deposits

4. **Normal Retina**
   - Preserved foveal contour, absence of fluid or edema

---

#### **About the Dataset**

Our dataset consists of **84,495 high-resolution OCT images** (JPEG format) organized into **train, test, and validation** sets, split into four primary categories:
- **Normal**
- **CNV**
- **DME**
- **Drusen**

Each image has undergone multiple layers of expert verification to ensure accuracy in disease classification. The images were obtained from various renowned medical centers worldwide and span across a diverse patient population, ensuring comprehensive coverage of different retinal conditions.

---

#### **Get Started**

- **Upload OCT Images**: Begin by uploading your OCT scans for analysis.
- **Explore Results**: View categorized scans and detailed diagnostic insights.
- **Learn More**: Dive deeper into the different retinal diseases and how OCT helps diagnose them.

---

#### **Contact Us**

Have questions or need assistance? [Contact our support team](#) for more information on how to use the platform or integrate it into your clinical practice.

    """)

elif app_mode == "About": 
    st.header("About")
    st.markdown("""
                #### About Dataset
                Retinal optical coherence tomography (OCT) is an imaging technique used to capture high-resolution cross sections of the retinas of living patients. 
                Approximately 30 million OCT scans are performed each year, and the analysis and interpretation of these images takes up a significant amount of time.
                (A) (Far left) choroidal neovascularization (CNV) with neovascular membrane (white arrowheads) and associated subretinal fluid (arrows). 
                (Middle left) Diabetic macular edema (DME) with retinal-thickening-associated intraretinal fluid (arrows). 
                (Middle right) Multiple drusen (arrowheads) present in early AMD. 
                (Far right) Normal retina with preserved foveal contour and absence of any retinal fluid/edema.

                ---

                #### Content
                The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (NORMAL,CNV,DME,DRUSEN). 
                There are 84,495 X-Ray images (JPEG) and 4 categories (NORMAL,CNV,DME,DRUSEN).

                Images are labeled as (disease)-(randomized patient ID)-(image number by this patient) and split into 4 directories: CNV, DME, DRUSEN, and NORMAL.

                Optical coherence tomography (OCT) images (Spectralis OCT, Heidelberg Engineering, Germany) were selected from retrospective cohorts of adult patients from the Shiley Eye Institute of the University of California San Diego, the California Retinal Research Foundation, Medical Center Ophthalmology Associates, the Shanghai First People’s Hospital, and Beijing Tongren Eye Center between July 1, 2013 and March 1, 2017.

                Before training, each image went through a tiered grading system consisting of multiple layers of trained graders of increasing exper- tise for verification and correction of image labels. Each image imported into the database started with a label matching the most recent diagnosis of the patient. The first tier of graders consisted of undergraduate and medical students who had taken and passed an OCT interpretation course review. This first tier of graders conducted initial quality control and excluded OCT images containing severe artifacts or significant image resolution reductions. The second tier of graders consisted of four ophthalmologists who independently graded each image that had passed the first tier. The presence or absence of choroidal neovascularization (active or in the form of subretinal fibrosis), macular edema, drusen, and other pathologies visible on the OCT scan were recorded. Finally, a third tier of two senior independent retinal specialists, each with over 20 years of clinical retina experience, verified the true labels for each image. The dataset selection and stratification process is displayed in a CONSORT-style diagram in Figure 2B. To account for human error in grading, a validation subset of 993 scans was graded separately by two ophthalmologist graders, with disagreement in clinical labels arbitrated by a senior retinal specialist.

                """)
#Prediction page
elif(app_mode=="Disease Identification"):
    st.header("🔍 Disease Identification")
    st.write("Upload a retinal OCT scan, or pick one of the sample images, then click **Predict**.")

    source = st.radio("Image source", ["Upload an image", "Use a sample image"], horizontal=True)
    image = None
    if source == "Upload an image":
        test_image = st.file_uploader("Upload your OCT image:", type=["jpg","jpeg","png"])
        if test_image is not None:
            image = Image.open(test_image)
    else:
        samples = list_samples()
        if samples:
            choice = st.selectbox("Sample image (the folder name is the true label)", list(samples.keys()))
            image = Image.open(samples[choice])
        else:
            st.info("No sample images found in the `samples` folder.")

    if image is not None:
        col_img, col_result = st.columns([1, 1])
        with col_img:
            st.image(image, caption="Input OCT scan", width="stretch")

        with col_result:
            if st.button("Predict", type="primary"):
                with st.spinner("Please Wait...."):
                    probs = model_prediction(image)
                result_index = int(np.argmax(probs))
                label = CLASS_NAMES[result_index]
                full_name, description, recommendation = CLASS_INFO[label]

                if label == "NORMAL":
                    st.success(f"Prediction: **{full_name}**")
                else:
                    st.error(f"Prediction: **{label} - {full_name}**")
                st.metric("Confidence", f"{probs[result_index]*100:.2f}%")

                st.subheader("Class probabilities")
                for name, p in sorted(zip(CLASS_NAMES, probs), key=lambda t: -t[1]):
                    st.write(f"{name}: {p*100:.2f}%")
                    st.progress(float(p))

                #Recomendation
                with st.expander("Learn More", expanded=True):
                    st.write(description)
                    st.markdown(recommendation)
