# 💡 Engineering Project: Hazard Detection for Toddlers

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
- [Docker](https://www.docker.com/)
- [Git](https://git-scm.com/)/

- 

### 🏗️ Installing
1. Clone the repository:
   ```bash
   git clone https://github.com/eitanste/Engineering-Project.git

2. Navigate to the project directory:
  ```bash
  cd Engineering-Project
3. Install Python dependencies:
```bash
  pip install -r requirements.txt

### 🏃‍♂️ Running the Code

Modify the input stream as needed by editing the deploy.py file.
Run the program using:
 ```bash
python deploy.py
