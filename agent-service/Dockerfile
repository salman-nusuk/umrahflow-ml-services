# Use the official Python image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies (psycopg[binary] ships prebuilt wheels, so no build deps needed)
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port
EXPOSE 8080

# Run the application (app.py exposes the FastAPI `app`)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
