{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "7871c8b0-6451-4a80-9567-e0a923ef70e0",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "8ad5dbcc-bf77-4c08-a866-4928129e3447",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "8b7d29a0-cb6c-4fc8-9479-7621c5a6f889",
   "metadata": {},
   "outputs": [],
   "source": [
    "import cv2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "887cd682-ee6c-4b93-860e-fc9bb28462ec",
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.models import load_model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "31e9773a-341e-41fb-b272-028aa00bcf3c",
   "metadata": {},
   "outputs": [],
   "source": [
    "from PIL import Image "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "8fadf0c7-f2ae-4ee1-bcf3-2d523801f908",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026-01-12 14:20:49.589 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "st.set_page_config(page_title=\"Traffic Sign Recognition\", page_icon=\"🚦\", layout=\"centered\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "a0646ef7-05ff-4f77-9a7f-16f42d3c252b",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026-01-12 14:21:17.267 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 14:21:17.670 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\LENOVO\\AppData\\Local\\Programs\\Python\\Python313\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2026-01-12 14:21:17.671 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 14:21:17.673 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 14:21:17.675 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 14:21:17.676 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 14:21:17.677 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "DeltaGenerator()"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "st.markdown(\"<h1 style='text-align:center;'>🚦 Traffic Sign Recognition</h1>\", unsafe_allow_html=True)\n",
    "st.markdown(\"<p style='text-align:center;'>Upload a traffic sign image to identify it using Deep Learning</p>\", unsafe_allow_html=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "08fa424f-894d-4c7d-bd00-103a36c5176c",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026-01-12 14:26:29.374 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 14:26:29.376 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 14:26:29.377 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 14:26:29.378 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 14:26:29.381 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 14:26:29.382 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 14:26:29.383 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "@st.cache_resource\n",
    "def load_cnn_model():\n",
    "    return load_model(\"model.h5\")\n",
    "\n",
    "try:\n",
    "    model = load_cnn_model()\n",
    "    model_loaded = True\n",
    "except Exception as e:\n",
    "    model_loaded = False"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "07d9b1cc-3578-4596-9f95-467baa5c4aa8",
   "metadata": {},
   "outputs": [],
   "source": [
    "CLASS_NAMES = [\n",
    "'Speed limit (20km/h)','Speed limit (30km/h)','Speed limit (50km/h)',\n",
    "'Speed limit (60km/h)','Speed limit (70km/h)','Speed limit (80km/h)',\n",
    "'End of speed limit (80km/h)','Speed limit (100km/h)','Speed limit (120km/h)',\n",
    "'No passing','No passing for vehicles over 3.5 metric tons','Right-of-way at the next intersection',\n",
    "'Priority road','Yield','Stop','No vehicles','Vehicles over 3.5 metric tons prohibited',\n",
    "'No entry','General caution','Dangerous curve to the left','Dangerous curve to the right',\n",
    "'Double curve','Bumpy road','Slippery road','Road narrows on the right','Road work',\n",
    "'Traffic signals','Pedestrians','Children crossing','Bicycles crossing','Beware of ice/snow',\n",
    "'Wild animals crossing','End of all speed and passing limits','Turn right ahead','Turn left ahead',\n",
    "'Ahead only','Go straight or right','Go straight or left','Keep right','Keep left',\n",
    "'Roundabout mandatory','End of no passing','End of no passing by vehicles over 3.5 metric tons'\n",
    "]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "32638e3e-d21f-4eea-bd44-9656ce3debee",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026-01-12 14:27:09.612 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 14:27:09.613 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 14:27:09.614 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 14:27:09.615 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 14:27:09.616 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 14:27:09.617 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 14:27:09.619 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 14:27:09.620 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 14:27:09.621 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "st.markdown(\"### 📤 Upload Traffic Sign Image\")\n",
    "uploaded_file = st.file_uploader(\"Choose an image (JPG / PNG)\", type=[\"jpg\", \"jpeg\", \"png\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "d1a8e132-a6da-4374-bd81-d9836a53811a",
   "metadata": {},
   "outputs": [],
   "source": [
    "if uploaded_file is not None:\n",
    "    image = Image.open(uploaded_file).convert('RGB')\n",
    "    st.image(image, caption=\"Uploaded Image\", use_column_width=True)\n",
    "\n",
    "    if st.button(\"🔍 Predict Traffic Sign\"):\n",
    "        if not model_loaded:\n",
    "            st.error(\"❌ Model not found. Please add 'model.h5' in the project folder.\")\n",
    "        else:\n",
    "            img = np.array(image)\n",
    "            img = cv2.resize(img, (32, 32))\n",
    "            img = img / 255.0\n",
    "            img = np.expand_dims(img, axis=0)\n",
    "\n",
    "            predictions = model.predict(img)\n",
    "            class_id = np.argmax(predictions)\n",
    "            confidence = np.max(predictions) * 100\n",
    "\n",
    "            st.success(f\"✅ Prediction: {CLASS_NAMES[class_id]}\")\n",
    "            st.info(f\"📊 Confidence: {confidence:.2f}%\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "a8bb4a55-7e49-4723-a269-70d865287d70",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026-01-12 14:29:28.786 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 14:29:28.787 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 14:29:28.788 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 14:29:28.789 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 14:29:28.790 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-01-12 14:29:28.791 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "DeltaGenerator()"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "st.markdown(\"---\")\n",
    "st.markdown(\"<p style='text-align:center;'>Developed using Deep Learning (CNN) | Streamlit UI</p>\", unsafe_allow_html=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "4c2de3ea-9da3-4a31-b376-a666613e9d2a",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
