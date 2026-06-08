# Use the official Python image
FROM python:3.12-slim

# System libs needed at runtime by opencv-python-headless (libglib) and
# the ONNX runtime used by fastmrz (libgomp). No build toolchain needed —
# all Python deps ship prebuilt wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port
EXPOSE 8080

# Run the application (app.py exposes the FastAPI `app`)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
