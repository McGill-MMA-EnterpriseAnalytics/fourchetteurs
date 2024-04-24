FROM ubuntu:latest AS ml_code

RUN apt-get update && \
    apt-get install -y git

    RUN mkdir /app && \
    cd /app && \
    git clone -b Deep-Learning-Clustering https://github.com/McGill-MMA-EnterpriseAnalytics/fourchetteurs.git

FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11 as base_image

FROM base_image AS train
COPY --from=ml_code /app/requirements.txt .
RUN pip install -r requirements.txt
COPY --from=ml_code /app/train-v0.py /app/train-v0.py
RUN python3 /app/train-v0.py

FROM base_image
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY --from=train /app/predictions.pkl .
COPY main-v0.py /app/main-v0.py
    

