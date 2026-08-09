# 🧬 OncoVision AI — Breast Cancer Detection from Ultrasound Images

A deep learning system that classifies breast ultrasound images into **Benign**, **Malignant**, or **Normal** using a custom Attention-based CNN, deployed as an interactive **Streamlit** web app.

> ⚠️ **Disclaimer:** This is a research/educational prototype. It is **not** a certified medical diagnostic tool. Always consult qualified healthcare professionals for actual diagnosis.

---

## 🗂️ Project Structure

```
.
├── breast-cancer.ipynb      # Data loading, EDA, preprocessing, and model training (Kaggle/Colab)
├── app.py                       # Streamlit web application for inference
├── breast_cancer_weights.weights.h5   # Trained model weights (required to run the app)
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

- **Source:** [Breast Ultrasound Images Dataset (BUSI)](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) on Kaggle
- **Classes:** `benign`, `malignant`, `normal`
- Each image comes with a corresponding segmentation mask highlighting the region of interest (tumor area).
- The dataset is imbalanced across classes, so **random oversampling** (via `imblearn`) is applied to balance class distribution before training.

---

## 🧠 Model Architecture

The core model is a custom **Attention-CNN**, built with the Keras Functional API:

1. **3 Convolutional blocks** (32 → 64 → 128 filters), each followed by max pooling.
2. **Channel Attention Module** after each block — uses global average pooling + a 1×1 convolution with sigmoid activation to learn per-channel importance weights, then multiplies them back into the feature map. This helps the model focus on the most diagnostically relevant regions (e.g., tumor borders).
3. **Flatten → Dense(256, ReLU) → Dropout(0.5)** for classification head.
4. **Dense(3, Softmax)** output layer for the three classes.

**Training configuration:**
- Input size: `224 × 224 × 3`
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Metric: Accuracy
- Data split: 80% train / 10% validation / 10% test (stratified)
- Batch size: 16
- Epochs: 15

A simpler baseline CNN (without attention) is also included in the notebook for comparison.

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
streamlit
tensorflow
Pillow
numpy
```

> 💡 If you also want to run the training notebook, you'll additionally need: `pandas`, `scikit-learn`, `imbalanced-learn`, `opencv-python`, `seaborn`, `matplotlib`, and `kaggle`.

### 4. Add the trained model weights
Place your trained weights file in the project root, named:
```
breast_cancer_weights.weights.h5
```
(Generate this by running the training notebook and saving the model weights, or download a pretrained copy if provided separately.)

---

## 🚀 Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### How to use it
1. **Upload** a breast ultrasound image (`.jpg`, `.jpeg`, or `.png`) on the upload page.
2. Click **🚀 Run AI Detection**.
3. View the **classification results page**, showing:
   - Confidence percentages for Benign / Malignant / Normal
   - A final AI decision with a color-coded result card (green = benign, red = malignant, blue = normal)
4. Click **← Analyze Another Image** to restart.

---

## 🏋️ Training Your Own Model

The full training pipeline is in the notebook and covers:
1. Downloading the BUSI dataset via the Kaggle API.
2. Building a labeled DataFrame of image/mask paths.
3. Exploratory data analysis (class distribution, sample image/mask visualizations).
4. Label encoding and oversampling to balance classes.
5. Train/validation/test split with `ImageDataGenerator` for on-the-fly rescaling.
6. Training a baseline CNN, then the Attention-CNN.
7. Evaluating with confusion matrices and classification reports.
8. Saving the final model (`model.save("my_model.keras")`) or exporting weights for the Streamlit app.

To reuse the weights in `app.py`, save just the weights after training:
```python
model.save_weights("breast_cancer_weights.weights.h5")
```

---

## 📈 Results

Model performance (accuracy, confusion matrix, precision/recall/F1) is generated at the end of the training notebook via `classification_report` and a Seaborn heatmap of the confusion matrix. Update this section with your own metrics once you've trained the model.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Model training | TensorFlow / Keras |
| Data processing | Pandas, NumPy, OpenCV |
| Class balancing | imbalanced-learn (RandomOverSampler) |
| Visualization | Matplotlib, Seaborn |
| Web app | Streamlit |
| Image handling | Pillow |

---

## 📄 License

Specify your license here (e.g., MIT).

## 🙏 Acknowledgments

- [BUSI Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) by Al-Dhabyani et al.
