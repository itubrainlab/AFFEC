# AFFEC
Advancing Face-to-Face Emotion Communication: A Multimodal Dataset (AFFEC)
# Multimodal Emotion Classification

## 📌 Overview
This project implements a **multimodal emotion classification pipeline** using **Eye Tracking, Facial Action Units (AU), Galvanic Skin Response (GSR), and Personality Traits**. The system extracts features, trains **AutoGluon-based machine learning models**, evaluates their performance with **5-Fold Cross-Validation**, and presents results in a **structured table format**.

This pipeline is designed for applications in **affective computing, human-computer interaction (HCI), and psychological research**.

---

## 📊 **Modality Breakdown**
This project allows flexible selection of **input modalities**:

| **Modality** | **Description** |
|-------------|----------------|
| **Eye Tracking** | Measures gaze, fixation, and pupil dilation. |
| **Facial Action Units (AU)** | Detects facial muscle movements and microexpressions. |
| **Galvanic Skin Response (GSR)** | Captures physiological arousal through skin conductance. |
| **Personality Traits** | Uses Big Five Personality Factors (OCEAN) for behavioral analysis. |

---

## 🚀 **How It Works**
### **1️⃣ Feature Extraction**
- Extracts statistical features (**mean, std, min, max**) from **Eye Tracking, AU, and GSR** data.
- Introduces **randomized personality traits** for robustness.
- Merges selected modalities dynamically.

### **2️⃣ Emotion Label Discretization**
- Converts **Perceived & Felt Arousal/Valence** into **Low, Medium, High** bins.

### **3️⃣ Model Training**
- Uses **AutoGluon TabularPredictor** for **automated model selection & hyperparameter tuning**.
- Trains models for **each emotion category**.

### **4️⃣ Results Evaluation**
- Performs **5-Fold Cross-Validation** to measure reliability.
- Displays **F1 Scores & Accuracy** in a structured table.

---

## 📌 **Project Structure**
📂 multimodal-emotion-classification │── 📄 README.md # Project documentation │── 📄 requirements.txt # Dependencies │── 📄 notebook.ipynb # Jupyter notebook containing code │── 📂 data # Folder to store input data (Eye, AU, GSR, Personality) │── 📂 models # Folder to save trained models │── 📂 results # Folder for storing classification results

---

## ⚙ **Installation & Setup**
### **1️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
2️⃣ Prepare Data (Download the data from [this link](https://doi.org/10.5281/zenodo.14794876))
Place your Eye Tracking, AU, GSR, and Personality data inside the /data directory.
Ensure the data follows the correct format.
3️⃣ Run Jupyter Notebook
Launch Jupyter Notebook and open:
jupyter notebook notebook.ipynb
Run each cell to train models and evaluate results.

🛠 Configuration
Modify the USE_MODALITIES dictionary in the notebook to enable/disable specific modalities:

python
Copy
Edit
USE_MODALITIES = {
    "eye": True,            # Use Eye Tracking data
    "action_units": True,   # Use Facial AU data
    "gsr": True,            # Use GSR data
    "personality": True     # Use Personality data
}
Set True to enable a modality.
Set False to disable a modality.
📈 Example Output
diff
Copy
Edit
## Classification Performance Using Multimodal Features (Eye, Facial Action Units, GSR, Personality)

| Metric     | Perceived Arousal         | Perceived Valence         | Felt Arousal              | Felt Valence              |
|------------|---------------------------|---------------------------|---------------------------|---------------------------|
| Best Model | XGBoost                   | XGBoost                   | LightGBMXT                | NeuralNetFastAI           |
| High       | 0.4565 ± 0.0160           | 0.2317 ± 0.0249           | 0.2692 ± 0.0401           | 0.4730 ± 0.0304           |
| Medium     | 0.3619 ± 0.0168           | 0.4306 ± 0.0168           | 0.4845 ± 0.0252           | 0.2938 ± 0.0270           |
| Low        | 0.4945 ± 0.0212           | 0.6104 ± 0.0148           | 0.6797 ± 0.0162           | 0.6133 ± 0.0249           |
| Macro Avg  | 0.4377 ± 0.0080           | 0.4242 ± 0.0152           | 0.4778 ± 0.0142           | 0.4600 ± 0.0194           |
| Accuracy   | 0.4415 ± 0.0085           | 0.5057 ± 0.0145           | 0.5680 ± 0.0158           | 0.5139 ± 0.0218           |

📌 Experiment Configurations
The notebook allows different modality combinations to compare performance.

Configuration	Eye Tracking	AU	GSR	Personality	Purpose
Full Multimodal	✅	✅	✅	✅	Uses all available data.
Eye Tracking Only	✅	❌	❌	❌	Evaluates eye-tracking data in isolation.
Facial AU Only	❌	✅	❌	❌	Uses only facial expressions for classification.
GSR Only	❌	❌	✅	❌	Evaluates physiological arousal.
Multimodal (No Personality)	✅	✅	✅	❌	Tests classification without personality factors.
🔧 Requirements
All required dependencies are listed in requirements.txt.

Dependencies
bash
Copy
Edit
pip install -r requirements.txt
Software Requirements
Python >=3.8
Jupyter Notebook
AutoGluon
Pandas, NumPy, SciKit-Learn
NeuroKit2 (for GSR processing)
Tabulate (for formatted result display)
