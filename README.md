# Human Eye Disease Detection System

A deep learning web app that classifies retinal **OCT (Optical Coherence Tomography)** scans into four classes:

| Class | Meaning |
|---|---|
| **CNV** | Choroidal Neovascularization |
| **DME** | Diabetic Macular Edema |
| **DRUSEN** | Drusen deposits (early AMD) |
| **NORMAL** | Healthy retina |

The model is a MobileNetV3-based CNN trained on the Kaggle dataset [Labeled Optical Coherence Tomography (OCT)](https://www.kaggle.com/datasets/anirudhcv/labeled-optical-coherence-tomography-oct), which has 84,495 images. It reaches about 97% validation accuracy.

## Project structure

```
app.py                          Streamlit web app (UI)
recommendation.py               Recommendations shown for each predicted disease
eye_disease_model.onnx          Trained model in ONNX format (used by the app)
Trained_eye_disease_model.h5    Trained Keras model
trained_eye_disease_model.keras Same model in .keras format
Traning_Model.ipynb             Training notebook
Model_Prediction.ipynb          Prediction notebook
tranning_history.pkl            Training history
samples/                        A few OCT images per class to try the app
```

## Run the app

The app runs the model with ONNX Runtime, so TensorFlow is not needed to run it (any recent Python works).

```bash
git lfs install
git clone https://github.com/Izmawarakhan/Human_Eye_Disease_Detection_System.git
cd Human_Eye_Disease_Detection_System
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501, go to **Disease Identification**, and upload an OCT image or pick a sample.

## Retraining

Download the dataset from Kaggle, extract it so that `train/`, `val/` and `test/` folders sit next to the notebook, then run `Traning_Model.ipynb`.

> For educational purposes only. This is not a medical diagnostic tool.
