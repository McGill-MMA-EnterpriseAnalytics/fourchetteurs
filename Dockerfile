FROM ubuntu:latest AS ml_code

RUN apt-get update && \
    apt-get install -y git

    RUN mkdir /app && \
    cd /app && \
    git clone -b Deep-Learing-Clustering https://github.com/McGill-MMA-EnterpriseAnalytics/fourchetteurs.git

FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11 as base_image

FROM base_image AS train
COPY --from=ml_code /app/fourchetteurs/app/requirements.txt .
RUN pip install -r requirements.txt
COPY --from=ml_code /app/fourchetteurs/app/train-v0.py /app/train-v0.py
# Copy the dataset into the container
COPY --from=ml_code /app/fourchetteurs/app/bank_marketing_dataset.csv /app/bank_marketing_dataset.csv
RUN python3 /app/train-v0.py
RUN ls -l /app/ 

FROM base_image AS fastapi_deployment
COPY --from=ml_code /app/fourchetteurs/app/requirements.txt .
RUN pip install -r requirements.txt
COPY --from=train /app/xgb_model_trained.pkl /app/xgb_model_trained.pkl
COPY --from=ml_code /app/fourchetteurs/app/main-v0.py /app/main-v0.py
WORKDIR /app
EXPOSE 80
CMD ["uvicorn", "main-v0:app", "--host", "0.0.0.0", "--port", "80"]

# Streamlit deployment stage: setup Streamlit application
FROM base_image AS streamlit_deployment
COPY --from=train /app/xgb_model_trained.pkl /app/
COPY --from=ml_code /app/fourchetteurs/app/predict-ui.py /app/predict-ui.py
WORKDIR /app
EXPOSE 8501
CMD ["streamlit", "run", "predict-ui.py"]