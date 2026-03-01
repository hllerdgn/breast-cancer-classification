import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data.csv")

# csv dosyasının okunup okunmadığının tespiti
print(df.head())
print("----------------------------------------------------------")
print("Sütunlar:", df.columns.tolist())
print("----------------------------------------------------------")

# M ve B etiketli verilerin sayısı
print(df["diagnosis"].value_counts())
print("----------------------------------------------------------")

# etiketlerin sayılara çevrilmesi (B=0, M=1)
le = LabelEncoder()
df["diagnosis"] = le.fit_transform(df["diagnosis"])

y = df["diagnosis"]

# tüm sütunlardan diagnosis ve id sütunlarının çıkarılması
X = df.drop(["diagnosis", "id"], axis=1)

# test verisine %20 lik bir alan ayırtarak modelin öğrenme parametrelerinin ayarlanması
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)


rf_model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
rf_model.fit(X_train, y_train)

print("Ağaç sayısı:", rf_model.n_estimators)
print("----------------------------------------------------------")

# öznitelik önem sırası
feature_importance = pd.Series(
    rf_model.feature_importances_, index=X.columns
).sort_values(ascending=False)

print("En önemli 10 öznitelik:")
print(feature_importance.head(10))
print("----------------------------------------------------------")

# test verilerinin y değerlerinin tahmin edilmesi
y_pred = rf_model.predict(X_test)

print("modelin değerlendirilmesi:")
print(
    f"Accuracy: {accuracy_score(y_test, y_pred)} ({accuracy_score(y_test, y_pred)*100:.2f}%)"
)
print(
    f"Precision: {precision_score(y_test, y_pred)} ({precision_score(y_test, y_pred)*100:.2f}%)"
)
print(
    f"Accuracy: {recall_score(y_test, y_pred)} ({recall_score(y_test, y_pred)*100:.2f}%)"
)
print(f"Accuracy: {f1_score(y_test, y_pred)} ({f1_score(y_test, y_pred)*100:.2f}%)")
print("----------------------------------------------------------")
