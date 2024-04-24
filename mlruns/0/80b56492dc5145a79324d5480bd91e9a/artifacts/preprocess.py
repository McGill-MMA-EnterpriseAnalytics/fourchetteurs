import pandas as pd
from sklearn import preprocessing, model_selection
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def preprocess_data(filepath):

    df = pd.read_csv(filepath)
    

    df['target'] = df['subscribed'].apply(lambda x: 1 if x == 'yes' else 0)
    df.drop('subscribed', axis=1, inplace=True)
    

    df.drop(['duration'], axis=1, inplace=True)  
    

    df = df.drop(columns=['euribor3m', 'emp.var.rate'])
    

    le = preprocessing.LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object' and len(df[col].unique()) <= 2:
            df[col] = le.fit_transform(df[col])
    

    df = pd.get_dummies(df)
    
    return df

def split_data(df):
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_val_test, y_train, y_val_test = model_selection.train_test_split(
        X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = model_selection.train_test_split(
        X_val_test, y_val_test, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

def scale_data(X_train, X_val, X_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)
    X_test = sc.transform(X_test)
    return X_train, X_val, X_test

def apply_smote(X_train, y_train):
    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
    return X_train_sm, y_train_sm

if __name__ == "__main__":
 
    filepath = 'bank_marketing_dataset.csv'
    

    df = preprocess_data(filepath)
    

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    X_train, X_val, X_test = scale_data(X_train, X_val, X_test)

    X_train_sm, y_train_sm = apply_smote(X_train, y_train)
