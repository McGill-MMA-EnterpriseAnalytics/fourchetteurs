import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import json  

# Load data
df_train = pd.read_csv('bank_marketing_dataset.csv')
df_train.drop(['duration', 'euribor3m', 'emp.var.rate','subscribed'], axis=1, inplace=True)

X = df_train

# random sample of 5 rows from X_train
X_train_transformed_df = X.sample(150, random_state=42)

json_test_data = json.dumps({"features": X_train_transformed_df.values.tolist()})

# Save the json file
with open('to predict/test_data.json', 'w') as f:
    f.write(json_test_data)

# Save as csv also:
X_train_transformed_df.to_csv('to predict/test_data.csv', index=False)