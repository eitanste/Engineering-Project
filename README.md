# 💡 Engineering Project: Hazard Detection for Toddlers

NOTICE THAT THIS IS THE LOCAL VERSION OF THE PROJECT. THE ONLINE STREAMING VERSION WHICH REQUIRES AN ACTIVE SERVER CONFIGURATION WILL NOT BE COVERED.


OUR WEBSITE: https://toddler-alert.com


![Project Cover Image](/media/project-cover-img.jpg)

## Table of Contents
- [The Team](#the-team)
- [Project Description](#project-description)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installing](#installing)
- [Running the Code](#running-the-code)
- [Folder Structure](#folder-structure)
- [Main Files](#main-files)
- [Deployment](#deployment)
- [Built With](#built-with)

## 👥 The Team 
**Team Members**
- [Eitan Stepanov](mailto:eitan.stein@example.com)
- [Tanya Fainstein](mailto:name@example.com)

**Supervisor**
- [Daniells Har Shalom](http://www.examplelab.com)

## 📚 Project Description
This project focuses on real-time detection of hazardous objects in environments with toddlers using YOLOv5. The system streams live video, identifies dangerous objects, and sends alerts via Telegram.

### Features and Functionalities:
- Real-time object detection using YOLOv5
- Hazardous object alerts via Telegram
- Live data streaming to a web dashboard
- User-friendly web interface

### Main Components:
- Object Detection Module
- Web Interface
- Notification System

### Technologies Used:
- **Languages:** Python, JavaScript
- **Frameworks:** Flask, OpenCV
- **Libraries:** PyTorch, Flask-SocketIO
- **Tools:** Git

## ⚡ Getting Started

These instructions will help you set up the project on your local machine for development and testing.

### 🧱 Prerequisites
- [Python 3.x](https://www.python.org/downloads/)
- [Git](https://git-scm.com/)
- Camera
- Cuda GPU

### 🏗️ Installing
1. Clone the repository:
   ```bash
   git clone https://github.com/eitanste/Engineering-Project.git

2. Navigate to the project directory:
   ```bash
   cd Engineering-Project/yolov5_deploy/


3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt


### 🏃‍♂️ Running the Code

Modify the input stream as needed by editing the deploy.py file.
Run the program using:

   ```bash
   python deploy.py

```

The system will begin processing the video stream and identifying hazardous objects.

### 📄 Main Files

deploy.py

This is the main entry point for running the project. It handles the video stream processing, object detection, and triggering notifications. The input stream is hardcoded within this file and can be modified manually.

consts.py

This file defines constant values used throughout the project, including configuration parameters, threshold values, and other constants.

main_manager.py

Manages the different components of the project, coordinating tasks like data processing, object detection, and communication between modules.

notification_manager.py

Responsible for managing notifications sent by the system, including setting up and sending alerts via Telegram when hazardous objects are detected.

object_detector.py

Implements the core object detection functionality using YOLOv5, processing input data and identifying objects of interest.

object_interaction.py

Handles interactions between detected objects, such as calculating proximity or determining if an object poses a risk.
