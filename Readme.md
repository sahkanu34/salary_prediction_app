# Salary Prediction App (Supervised Learning Project)

<img src="https://github.com/sahkanu34/salary_prediction_app/blob/main/app.png">

# The app is already been deployed
[Open Live Project](https://salarypredictionapp34.streamlit.app/)

# Project Overview
This is a Machine Learning web application that predicts salary based on various professional attributes using Random Forest Regression. The application leverages supervised learning techniques to provide accurate salary estimates.

# Features
- Predict salary using multiple input features
- Interactive web interface with Streamlit
- Machine Learning model using Random Forest Regressor
- Easy to use and deploy

<img src="https://github.com/sahkanu34/salary_prediction_app/blob/main/sal_app.png">

## Prerequisites
- Python 3.8+
- pip
- Docker

## Installation
```bash
pip install -r requirements.txt
```

## Pull the Image from my Dockerhub
```bash
docker pull sahkanu37/salary_prediction
```
## Go to my Dockerhub Account
```bash
https://hub.docker.com/r/sahkanu37/salary_prediction
```

### Local Setup
1. Clone the repository
```bash
git clone https://github.com/yourusername/salary-prediction-app.git
cd salary-prediction-app
```
### Docker Setup
1. Build the Docker Image
```bash
docker build -t salary-prediction-app .
docker image ls
```
2. Run the container
```bash
docker run -d --rm --name "salary" -p 8501:8501 <Image Id>
docker ps
docker stop <container_name>
```
3. Access your application (Through Docker)
```bash
localhost:8501
```
4. Access your application (Through Streamlit)
```bash
streamlit run app.py
```

# Thank your for your support:
<img src="https://media.istockphoto.com/id/1397892955/photo/thank-you-message-for-card-presentation-business-expressing-gratitude-acknowledgment-and.jpg?s=612x612&w=0&k=20&c=7Lyf2sRAJnX_uiDy3ZEytmirul8pyJWm4l2fxiUtdvk=">
