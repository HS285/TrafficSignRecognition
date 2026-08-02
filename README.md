# Traffic Sign Recognition Web App 🚦

A deep learning-powered web application that recognizes and classifies traffic signs from uploaded images. Built with **Streamlit** for the interactive user interface and **TensorFlow/Keras** for the classification engine.

---

## 🌟 Key Features

* **Instant Classification**: Upload any traffic sign image, and the model will predict the sign class instantly.
* **Streamlit Interface**: Clean, modern, responsive user interface.
* **Pre-trained CNN Model**: Uses a Convolutional Neural Network (CNN) model (`model.h5`) trained on the GTSRB (German Traffic Sign Recognition Benchmark) dataset to recognize up to 43 different types of traffic signs.

---

## 🛠️ Project Structure

* `streamlit_app.py`: Main python script containing the Streamlit application code.
* `model.h5`: Trained Keras Convolutional Neural Network (CNN) model.
* `requirements.txt`: Python package dependencies.
* `.streamlit/`: Configurations for the web deployment.

---

## 💻 Tech Stack & Dependencies

* **Frontend Framework**: Streamlit
* **Deep Learning Framework**: TensorFlow / Keras
* **Image Processing**: OpenCV (opencv-python-headless), Pillow
* **Numerical Operations**: NumPy

---

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/hitesh2805/TrafficSignRecognition.git
cd TrafficSignRecognition
```

### 2. Set Up a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
Install all the required python libraries using:
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
Start the local development server:
```bash
streamlit run streamlit_app.py
```
*Once running, open your web browser and navigate to the address shown in the terminal (usually `http://localhost:8501`).*

---

## 🤖 Model Details

The model takes an input image of a traffic sign, preprocesses it (resizing and normalizing), and feeds it into a Convolutional Neural Network (`model.h5`). The system supports the 43 standard classes of traffic signs:
1. Speed limits (20km/h up to 120km/h)
2. Regulatory signs (Stop, Yield, No Entry, Turn Direction, etc.)
3. Warning signs (General Caution, Road Work, Pedestrians, Bicycles, Slippery Road, etc.)