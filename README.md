# CCTV Anomaly Detection System

[Live Demo](https://anomaly-detection-from-cctv-footage.onrender.com)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies](#technologies)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Results and Evaluation](#results-and-evaluation)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Project Overview

This project is an end-to-end machine learning system that detects anomalies in CCTV video footage. It aims to identify suspicious or abnormal human activities such as abuse, arson, arrest, and assault through multi-class classification of video frames. The system extracts frames from video streams, processes them through pretrained models, and classifies each frame for real-time monitoring and public safety assistance.

---

## Features

- Upload CCTV videos via a user-friendly web interface (drag & drop or file select)
- Detect and classify anomalies using a pretrained machine learning model backend
- Visualize anomaly categories with confidence scores and detailed probability bars
- System status display showing model health and loaded categories
- Responsive single-page frontend with embedded CSS and JavaScript
- REST API based on Flask for backend processing
- Deployment-ready, with live demo hosted on Render.com

---

## Technologies

- Python 3.10+
- Flask & Flask-CORS (backend API)
- Scikit-learn, joblib (model training and serialization)
- OpenCV (video frame extraction and processing)
- NumPy, pandas, matplotlib, seaborn (data handling and visualization)
- Gunicorn (production WSGI server)
- HTML, CSS, JavaScript (single-page frontend UI)
- Deployment with Render.com

---

## Installation

### Prerequisites

- Python 3.10 or higher installed
- Git installed

### Clone the Repository
git clone https://github.com/Gautham-Ashok/Anomaly-Detection-from-CCTV-footage.git
cd Anomaly-Detection-from-CCTV-footage


### Create and Activate Virtual Environment
python -m venv anomaly_venv
Windows
anomaly_venv\Scripts\activate

macOS/Linux
source anomaly_venv/bin/activate

### Install Dependencies

pip install -r requirements.txt

---

## Usage

### Running Locally

Start the Flask backend server:

python api/app.py

Open [http://localhost:5000](http://localhost:5000) in your browser.

### Using the Web UI

- Upload CCTV video files (supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`)
- Wait for the model to analyze and display detection results with confidence scores
- View system status for model health and loaded categories

---

## Model Training

If you want to retrain or improve the model:

1. Extract frames from videos:

python extracted_frames.py


2. Train model on prepared data:

python train_model.py

3. Evaluate model:

python scripts/evaluate_model.py

Trained models are saved as `.joblib` files under `data/processed/models/`.

---

## Deployment

This project is deployed live on [Render.com](https://anomaly-detection-from-cctv-footage.onrender.com).

- The repository contains a `Procfile` and `requirements.txt` to facilitate deployment
- Backend runs with Gunicorn and Flask in production mode on Render
- Frontend can be served via Flask backend or any static hosting service
- Ensure frontend API URLs point to deployed backend domain

---

## Project Structure

/api # Flask backend code
/data # Data files & trained models
/frontend # Frontend (index.html with embedded CSS & JS)
/scripts # Helper scripts for training and evaluation
/requirements.txt # Python dependencies
/Procfile # Deployment configuration for Render
/README.md # Project documentation (this file)


---

## Results and Evaluation

The model performance on test data includes:
- Overall Accuracy: ~81%
- Weighted F1 Score: ~80%
- Per-class performance metrics for categories such as normal, road accidents, robbery, and abuse
- Confusion matrices and other visual evaluation artifacts generated during training

---

## Future Work

- Real-time detection with IP cameras or webcams
- SMS/email alert integration for detected anomalies
- Edge deployment on lightweight devices such as Raspberry Pi
- Integration of more advanced anomaly detection models like YOLOv8
- Expansion of anomaly categories and dataset size for robustness

---

## Contributing

Contributions are welcome! Feel free to fork, submit issues, or create pull requests for bugfixes, features, or improvements.

---

## License

This project is released under the MIT License.

---

## Contact

For questions, support, or collaborations, please open an issue or contact the repository owner.

---

**Live Demo:** [https://anomaly-detection-from-cctv-footage.onrender.com](https://anomaly-detection-from-cctv-footage.onrender.com)

---

*Developed with ❤️ for safer communities.*
