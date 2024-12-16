# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install numpy pandas streamlit scikit-learn plotly

# Make port 8501 available to the world outside this container
EXPOSE 8501:8501

# Define environment variable to ensure output is sent to terminal
ENV PYTHONUNBUFFERED=1

# Run streamlit when the container launches
CMD ["streamlit", "run", "app.py"]