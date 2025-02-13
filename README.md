# 🚀 **Autonomous Driving & Road Sign Detection using Computer Vision and LiDAR**  

## 📌 **Overview**  
This repository contains a collection of scripts for **autonomous vehicle navigation, road sign detection, LiDAR-based obstacle avoidance, and deep learning-based steering control**. The project integrates **image thresholding, HOG+SVM classification, CNN-based recognition, PID-controlled driving, and LiDAR-based navigation** in a simulated environment using **Webots, OpenCV, and TensorFlow**.  

## 🏗 **Project Structure**  

```
📂 project-root
├── 📂 dataset/                     # Folder containing dataset files
├── 📂 models/                      # Folder for trained models
├── 📂 logs/                        # Folder for storing logs and training history
│
├── 📜 1_create_dataset.py          # Captures and stores driving data for training
├── 📜 1_image_threshold.py         # Color thresholding for road sign detection
├── 📜 1_lidar_manual.py            # Manual LiDAR-based driving with keyboard/gamepad
│
├── 📜 2_camera_pid.py              # Camera-based PID steering control for navigation
├── 📜 2_hog_svm.py                 # HOG + SVM classifier for road sign recognition
├── 📜 2_lidar_automated.py         # Automated LiDAR-based obstacle avoidance
│
├── 📜 3_train.py                   # Train a CNN model for steering prediction
├── 📜 3_train_reg.py               # Train a regression-based steering model
│
├── 📜 4_run_model.py               # Runs the trained model for autonomous driving
│
├── 📜 driving_inputs.py            # Xbox/Keyboard input handler for manual driving
├── 📜 preprocessing.py             # Prepares dataset (cropping, resizing, augmentation)
│
├── 📒 1_hog.ipynb                  # HOG feature visualization
├── 📒 2_dataset_insights.ipynb     # Exploratory Data Analysis on collected dataset
├── 📒 3_cnn.ipynb                  # CNN-based classification model for traffic signs
├── 📒 5_explain.ipynb              # Model explainability and feature analysis
```

---

## 🖥️ **Installation & Setup**  

### 📌 **Prerequisites**  
Ensure you have **Python 3.8+** and the following dependencies installed:  

```bash
pip install numpy opencv-python matplotlib scikit-learn scikit-image webots pandas tensorflow inputs pynput
```

### 🔧 **Running the Scripts**  

#### 🎥 **Create Driving Dataset**  
```bash
python 1_create_dataset.py
```
- Captures images and steering data while driving manually for training an AI model.  

#### 🚦 **Road Sign Detection using Color Thresholding**  
```bash
python 1_image_threshold.py
```
- Reads an image, converts it to HSV, and applies color thresholding to detect specific sign colors.  

#### 🏎 **Camera-based PID Control for Autonomous Navigation**  
```bash
python 2_camera_pid.py
```
- Uses Webots simulation and a **PID controller** for lane keeping and smooth steering.  

#### 🔍 **HOG + SVM Classifier for Road Signs**  
```bash
python 2_hog_svm.py
```
- Extracts **Histogram of Oriented Gradients (HOG)** features and trains an **SVM** model for sign classification.  

#### 🛑 **Manual LiDAR-based Obstacle Avoidance**  
```bash
python 1_lidar_manual.py
```
- Uses **LiDAR sensor data** to create a 2D occupancy grid and allows manual driving with a **keyboard or Xbox controller**.  

#### 🚗 **Automated LiDAR-based Navigation**  
```bash
python 2_lidar_automated.py
```
- Implements an automated **obstacle avoidance algorithm** using LiDAR data, steering the vehicle around obstacles dynamically.  

#### 🎯 **Train a Deep Learning Model for Steering Control**  
```bash
python 3_train.py
```
- Trains a **Convolutional Neural Network (CNN)** model to predict steering angles from images.  

#### 📈 **Train a Regression Model for Steering**  
```bash
python 3_train_reg.py
```
- Trains a **regression-based model** for predicting steering angles from images.  

#### 🏁 **Run the Trained Model for Autonomous Driving**  
```bash
python 4_run_model.py
```
- Uses a **pretrained CNN or regression model** to control the vehicle autonomously.  

---

## 📊 **Machine Learning Approach**  

- **HOG + SVM**: Extracts **HOG features** and uses an **SVM** for robust classification of road signs.  
- **CNN-based Classification**: Uses a **Convolutional Neural Network** to classify traffic signs based on image data.  
- **LiDAR-based Navigation**: Processes **LiDAR point cloud data** to determine safe paths for automated driving.  
- **End-to-End Deep Learning for Steering**: Trains a **CNN-based model** that learns to drive from human demonstrations.  

---

## 📁 **Dataset**  
We use the **[Road Sign Detection Dataset](https://www.kaggle.com/andrewmvd/road-sign-detection)**, which includes:  

✔ 🚦 **Traffic Lights**  
✔ 🚏 **Stop Signs**  
✔ 🛑 **Speed Limit Signs**  
✔ 🚸 **Crosswalks**  

Additionally, the **driving dataset** is created using `1_create_dataset.py`, where images and corresponding steering angles are recorded during manual driving.  

---

## 📌 **Key Features**  

✔ **Real-time road sign detection**  
✔ **PID-based vehicle control for lane keeping**  
✔ **HOG + SVM classification for traffic sign recognition**  
✔ **CNN for sign recognition and deep learning-based steering**  
✔ **LiDAR-based automated navigation for obstacle avoidance**  
✔ **Manual driving with Xbox/Keyboard input**  
✔ **Deep Learning-based end-to-end steering prediction**  

---

## 🛠 **Contribution**  
Feel free to fork this repository, submit issues, or open pull requests! Contributions are always welcome.  

## 🏆 **Acknowledgments**  
Special thanks to the **Webots community, OpenCV, and TensorFlow** for their excellent tools.  

