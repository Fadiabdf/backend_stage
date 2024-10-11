# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory in the container
WORKDIR /myproject

# Copy the entire project (including requirements.txt) into the container
COPY . /myproject

# Download spaCy model
RUN python -m spacy download fr_core_news_sm

# Install the dependencies
RUN pip install --no-cache-dir -r /myproject/myproject/requirements.txt

# Expose the port the app runs on
EXPOSE 8080

# Set environment variables
ENV DJANGO_SETTINGS_MODULE=myproject.settings

# Run the Django app using Gunicorn
CMD ["gunicorn", "--workers=2", "--timeout=300", "myproject.wsgi:application", "--bind", "0.0.0.0:8080"]
