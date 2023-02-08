FROM python:3.8.12-slim-buster

# Install wget and required pip libraries
RUN apt-get update &&\
    apt-get install -y --no-install-recommends wget &&\
    rm -rf /var/lib/apt/lists/* &&\
    pip install --no-cache-dir \
      transformers[torch] uvicorn fastapi jinja2 python-multipart

RUN pip install transformers==4.26.0 &&\
    pip install beautifulsoup4 &&\
    pip install hangul_utils &&\
    pip install html5lib &&\
    pip install datetime &&\
    pip install torch==1.9.1

# copies the app files to the docker image
COPY app/ app/

# sets the working directory to the app folder
WORKDIR /app

# expose port for app
EXPOSE 8000

# runs our application at the start of the docker image
CMD ["python", "main.py"]