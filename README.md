# INSY 695 Final Project
## Group 3: Fourchetteurs

Data source: https://www.kaggle.com/datasets/berkayalan/bank-marketing-data-set

### Team Members
- Dhevin De Silva
- Keani Schuller
- Olivier Larochelle
- Tiffany Lagarde
- Valentin Najean

### Project Overview
This project aims to enhance the effectiveness of direct marketing campaigns for banks using machine learning. By leveraging a dataset from the University of California Irvine’s Machine Learning Repository, we explored innovative strategies to attract new clients and retain existing ones, transitioning from traditional to data-driven marketing approaches.

### Model Development Insights
- Classification Models: These models predict potential subscriber behavior, enhancing personalization of marketing campaigns. Techniques include feature selection using Random Forest and performance evaluation through accuracy and F1 score metrics.
- Clustering Models: Various techniques were assessed, with the best performers identified based on silhouette, Calinski-Harabasz, and Davies-Bouldin scores. This approach segments customers by behavior for targeted marketing strategies.
### Advanced Techniques
- Causal Inference: Introduced to understand the impact of marketing actions, such as campaign frequency and interest rates, on customer decisions.
- Model Explainability: Tools like SHAP and LIME provide insights into factors influencing model predictions, offering transparency in machine learning processes.
- Hyperparameter Tuning and AutoML: Techniques like Bayesian optimization and TPOT optimize models for peak performance.
### Production and Deployment
- MLflow: Manages the entire machine learning lifecycle, including experimentation, model tuning, and deployment. It tracks experiments to streamline training, parameter tuning, and evaluation processes.
- Docker: Used for production, Docker containers ensure that models run consistently across different systems, simplifying deployment cycles and enhancing operational reliability.
### Interactive Application
An interactive Streamlit application allows real-time interaction with the model's capabilities, accessible here: https://ui-classfication.azurewebsites.net/
