FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the application code
COPY app/ ./app/
COPY mnist_cnn.pth ./  
# Include the pretrained model file (train.py must be run prior to building)

# Expose the port for Streamlit (default is 8501)
EXPOSE 8501

# Set the working directory to app and run Streamlit
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
