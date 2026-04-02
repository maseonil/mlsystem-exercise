import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import os
import warnings
import sys

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # 1. Optimasi pembacaan file
    file_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_pca.csv")
    data = pd.read_csv(file_path)

    X = data.drop("Credit_Score", axis=1)
    y = data["Credit_Score"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    
    # Ambil contoh input untuk MLflow
    input_example = X_train.iloc[:5]

    # Parameter dari argumen atau default yang lebih ringan
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 505 # Default diturunkan ke 100 agar cepat
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 50     # Kedalaman dikurangi

    # 2. Aktifkan Autolog agar tidak perlu log manual satu per satu
    mlflow.sklearn.autolog()

    with mlflow.start_run():
        # 3. Gunakan n_jobs=-1 untuk menggunakan semua core CPU
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            n_jobs=-1, 
            random_state=42
        )
        
        # Latih model (CUKUP SEKALI)
        model.fit(X_train, y_train)

        # 4. Metrik akan otomatis tercatat berkat autolog()
        # Jika ingin log manual akurasi saja:
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)