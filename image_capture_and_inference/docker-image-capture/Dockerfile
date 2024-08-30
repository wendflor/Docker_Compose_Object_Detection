FROM stereolabs/zed:4.1-py-runtime-jetson-jp6.0.0 

# Create the app directory
RUN mkdir /app

# Set the working directory
WORKDIR /app

# Upgrade pip
RUN pip3 install --upgrade pip

# Copy the requirements file into the app directory
COPY requirements.txt /app

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . /app

# Run your application
CMD ["python3", "zed_image_capture.py"]
