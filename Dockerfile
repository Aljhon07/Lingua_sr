# Use Python 3.12 Slim as the base image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose port 80 (or any other port your application listens to)
EXPOSE 80

# Command to run the application
CMD ["python", "app.py"]  # Change "app.py" to your application's entry point
