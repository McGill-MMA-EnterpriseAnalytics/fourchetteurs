import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib

# Load data
df_train = pd.read_csv('bank_marketing_dataset.csv')

# Encode binary outcome
df_train['target'] = (df_train['subscribed'] == 'yes').astype(int)
df_train.drop('subscribed', axis=1, inplace=True)

# Drop columns not needed
df_train.drop(['duration', 'euribor3m', 'emp.var.rate'], axis=1, inplace=True)

# Identify columns by type
categorical_features = df_train.select_dtypes(include=['object']).columns.tolist()
binary_features = [col for col in categorical_features if len(df_train[col].unique()) <= 2]
high_cardinality_features = [col for col in categorical_features if len(df_train[col].unique()) > 2]

# Preprocessing steps for categorical data
binary_transformer = Pipeline(steps=[
    ('label_encoder', OneHotEncoder(drop='if_binary'))])

high_card_features_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('bin', binary_transformer, binary_features),
        ('high_card', high_card_features_transformer, high_cardinality_features)],
    remainder='passthrough')

# Define the model pipeline
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler(with_mean=False)),  # Adjust scaler to handle sparse data
    ('smote', SMOTE(random_state=42)),
    ('classifier', xgb.XGBClassifier(
        colsample_bytree=0.8,
        learning_rate=0.2,
        max_depth=7,
        n_estimators=100,
        random_state=42,
        subsample=1.0))
])

# Split data
X = df_train.drop('target', axis=1)
y = df_train['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

# Train the pipeline
pipeline.fit(X_train, y_train)

# Evaluate the model
print("Training complete. Model accuracy:", pipeline.score(X_test, y_test))

# Save the pipeline
joblib.dump(pipeline, '/app/xgb_model_trained.pkl')

print(X_train.columns)