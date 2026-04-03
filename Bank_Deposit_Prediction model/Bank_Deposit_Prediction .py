# -------------------------------------------------------
# Bank Deposit Prediction - Logistic Regression
# -------------------------------------------------------

# Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Step 2: Load & Explore Dataset
df = pd.read_csv("train.csv")
print("Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum().sum(), "→ None!")
print("\nClass Balance:\n", df['y'].value_counts())

# Step 3: Convert Text Columns to Numbers
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

# Step 4: Split into Input (X) and Output (y)
X = df.drop('y', axis=1)
y = df['y']

# Step 5: Scale the Features (makes model more accurate)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Step 6: Split Data - 80% Train, 20% Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTrain: {len(X_train)} rows | Test: {len(X_test)} rows")

# Step 7: Train the Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("\nModel Trained Successfully!")

# Step 8: Predictions & Accuracy
y_pred = model.predict(X_test)
print(f"\nAccuracy : {round(accuracy_score(y_test, y_pred) * 100, 2)}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 9: Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix - Nikhil Sabat")
plt.colorbar()
plt.xticks([0,1], ['No','Yes'])
plt.yticks([0,1], ['No','Yes'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i][j], ha='center', va='center', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("result.png", dpi=150)
plt.show()
print("\nGraph saved as result.png")
