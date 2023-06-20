# Use an official Python runtime as the base image
FROM python:3.9

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install any required dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt


# Copy the Python script into the container
COPY your_script.py .

# Set the command to run when the container starts
CMD ["python", "your_script.py"]
