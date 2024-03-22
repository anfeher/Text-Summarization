FROM python:3.12.1-slim-bullseye

WORKDIR /app

RUN apt update -y

COPY README.md README.md
COPY src src
COPY setup.py setup.py
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY artifacts/model_trainer artifacts/model_trainer
COPY config config
COPY params.yml params.yml
COPY main.py main.py
COPY app.py app.py

CMD ["python3","app.py"]