# INSY 695 Final Project
## Group 3: Fourchetteurs

### Project Overview
This project aims to enhance the effectiveness of direct marketing campaigns for banks using machine learning. By leveraging a dataset from the University of California Irvine’s Machine Learning Repository, we explored innovative strategies to attract new clients and retain existing ones, transitioning from traditional to data-driven marketing approaches.

### Business Context
The banking environment is competitive and traditionally rigid. Our objective is to use bank marketing data to personalize marketing strategies for banking institutions. This would allow for more efficient allocation of their resources and higher conversion rates, ultimately increasing profitability. Our dataset consists of 45,211 rows and 21 columns, including both numerical and categorical variables. It details the marketing campaigns conducted via phone calls of a Portuguese bank to get clients to subscribe a term deposit. To do this, we will be using two types of models: classification and clustering.

### Modelling Context
Classification Models: We predict the likelihood of a client subscribing to a term deposit, focusing on those with the highest likelihood to convert. This approach helps in personalizing marketing efforts and reducing costs. Our goal for our predictive classification is to predict client subscription to term deposits.

Clustering Models: We segment the bank's clients based on various characteristics, allowing for tailored marketing strategies that are more effective in engaging distinct client groups. Our segmentation goal is to identify distinct client segments for targeted marketing.

### Hypotheses
*"Classification techniques will accurately predict subscription likelihood based on client traits and previous campaign interactions."*

*"Clustering algorithms will reveal distinct groups within the client base, highlighting various patterns and characteristics for targeted marketing."*

### Model Building

We firstly took preprocessing steps to encode categorical variables, and normalize numerical variables. We then split the dataset into training and test sets to ensure a fair evaluation of model performance.
We implemented different classification algorithms, including Logistic Regression, Random Forest, XGBoost, CatBoost, AdaBoost, and Neural Networks, to predict client subscription to term deposits.
We then explored clustering techniques such as K-Means, GMM, DBSCAN, Mean Shift, and Hierarchical clustering to segment the client base.

### Modelling Results
Our models demonstrated varied performance, with the following highlights:
* Classification Models: XGBoost achieved the highest accuracy, closely followed by CatBoost and AdaBoost. Logistic Regression and Neural Networks provided valuable insights despite lower accuracy.
 * Clustering Models: Mean Shift, KNN, and GMM were identified as the best performing clustering techniques. They each had a different number of clusters: Mean Shift had four, GMM had two, and KNN had five.


### Interpretations
The success of XGBoost and CatBoost underscores the effectiveness of gradient boosting techniques in handling complex datasets with mixed variable types. Our feature importance analysis highlighted key predictors for subscription likelihood, including number of employees, phoning landlines over cellphones, campaigning during the month of May. Additionally using SHAP showed that the less often a client was contacted, the more likely they are to accept a term deposit. Our clustering results reveal distinct client segments, suggesting targeted marketing strategies could significantly improve campaign outcomes. The successful clusters in each model included non-subscribed users that could be the bank's target demographic. This shows the importance of the economic situation rather than the demographic details of clients.

### Conclusion
This project illustrates the potential of machine learning to transform bank marketing strategies through predictive modeling and client segmentation. By accurately predicting client subscription likelihood and identifying distinct client segments, banks can tailor their marketing efforts more effectively, leading to improved conversion rates and enhanced customer satisfaction.
