# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory in the container
WORKDIR /myproject

# Copy the current directory contents into the container
COPY . /myproject

# Debugging: list files to check if requirements.txt exists
RUN ls -al /myproject

# Install the dependencies
RUN pip install --no-cache-dir -r  requirements.txt
RUN python -m spacy download fr_core_news_sm


# Expose the port the app runs on
EXPOSE 8080

# Set environment variables
ENV DJANGO_SETTINGS_MODULE=myproject.settings

# Run the Django app using Gunicorn (you can use 'waitress' if you prefer)
CMD ["gunicorn","--workers=2","--timeout=300","myproject.wsgi:application", "--bind", "0.0.0.0:8080"]
