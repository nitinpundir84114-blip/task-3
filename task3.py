import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print("Loading Dataset...")

df=pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print(df.head())

print("\nDataset Shape :",df.shape)

# Remove Customer ID
df.drop("customerID",axis=1,inplace=True)

# Convert TotalCharges to Numeric
df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors="coerce")

# Fill Missing Values
df.fillna(0,inplace=True)

# Encode Categorical Columns
encoder=LabelEncoder()

for col in df.columns:
    if df[col].dtype=="object":
        df[col]=encoder.fit_transform(df[col])

# Features and Target
X=df.drop("Churn",axis=1)
y=df["Churn"]

# Train Test Split
X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Model...")

model=RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train,y_train)

# Prediction
y_pred=model.predict(X_test)

print("\nAccuracy :",accuracy_score(y_test,y_pred))

print("\nClassification Report")
print(classification_report(y_test,y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test,y_pred))

# Test One Customer
sample=X_test.iloc[[0]]

prediction=model.predict(sample)

if prediction[0]==1:
    print("\nPrediction : Customer Will Leave")
else:
    print("\nPrediction : Customer Will Stay")