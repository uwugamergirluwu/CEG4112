# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

# Copy requirements first (optimization for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy application code
COPY app.py .

#Expose port
EXPOSE 5000

#Command to run the application with Waitress
CMD ["waitress-serve", "--port=5000", "app:app"]