#  CCTV Anomaly Detection using Machine Learning

This project implements a machine learning-based anomaly detection system to identify suspicious or abnormal activities captured in CCTV surveillance footage. The goal is to assist in real-time monitoring and safety enforcement by classifying frames from security videos into categories of anomalies such as abuse, arson, arrest, and assault.

---

##  Objective

To detect and classify abnormal human activities in CCTV footage using pretrained machine learning models. The system processes video data, extracts frames, and uses deep learning models for multi-class classification of anomalies.

---

##  Project Structure
cctv-anomaly-detection/
│
├── dataset/
│ ├── Anomaly Part 1/
│ │ └── Anomaly-Videos-Part-1/
│ │ ├── Abuse/
│ │ ├── Arson/
│ │ ├── Arrest/
│ │ └── Assault/
│ ├── Anomaly Part 2/
│ ├── Anomaly Part 3/
│ └── Anomaly Part 4/
│
├── extracted_frames.py # Extracts frames from videos
├── train_model.py # Trains ML model on the dataset
├── predict.py # Performs inference on new frames
├── utils.py # Helper functions
├── requirements.txt
└── README.md

---

##  How It Works

1. **Frame Extraction**  
   - Videos are read from multiple folders categorized by anomaly type.
   - Frames are extracted every few seconds and saved for training/testing.
   - Handles nested folders with video clips.

2. **Model Training**  
   - CNN-based models like ResNet50, DenseNet, and eXception are used.
   - Models are fine-tuned using transfer learning.
   - Input: extracted frames labeled by their source folder name.

3. **Prediction**  
   - Trained models are used to classify new video frames.
   - Predicts whether the frame is normal or an anomaly class.

4. **Evaluation**  
   - Accuracy, confusion matrix, and F1 score are used for performance evaluation.

---

## Technologies Used

- **Python 3.10+**
- **TensorFlow / Keras**
- **OpenCV**
- **NumPy**
- **Matplotlib**
- **Scikit-learn**

---

##  ML Models Used

- **ResNet50**
- **DenseNet121**
- **eXception**
- Custom layers added for anomaly classification.
- Optimized using categorical crossentropy and Adam optimizer.

---

##  How to Run

1. **Clone the repository**
   ```bash

   git clone https://github.com/yourusername/cctv-anomaly-detection.git
   cd cctv-anomaly-detection
Install required packages

bash
Copy
Edit
pip install -r requirements.txt
Extract frames from video dataset

bash
Copy
Edit
python extracted_frames.py
Train the model

bash
Copy
Edit
python train_model.py
Run inference on new data

bash
Copy
Edit
python predict.py Output
Extracted frames saved in frames/

Trained model saved as .h5 file

Inference results printed and optionally saved with class labels

🔮 Future Scope
Add real-time detection via webcam or IP camera

Deploy on Raspberry Pi for edge applications

Integrate with alert system via SMS/email

Use advanced models like YOLOv8 for faster detection

