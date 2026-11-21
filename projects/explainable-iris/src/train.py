# Imports

from pathlib import Path

import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():

    # Load the Iris Dataset

    iris = load_iris() #1

    X = iris.data
    y = iris.target

#features_names = iris.features_names
    class_names = iris.target_names

    print("Iris Dataset Loaded.")
    print(f"Samples: {len(X)}")
    print(f"Features: {iris.feature_names}")
    print(f"Classes: {class_names}")

    # Train/Test Split (Split Dataset) #2

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Training Samples: {len(X_train)}")
    print(f"Testing Samples:  {len(X_test)}")

    # Create Model #3

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    
    # Train Random Forest
    
    model.fit(X_train, y_train)
    
    # Evaluate Model
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred) #4
    
    print(f"\nAccuracy: {accuracy:.2%}")

    print(f"\nClassification Report:") #5
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            zero_division=0
        )
    )

    # Save the Trained Model

    model_dir = Path("models") # Gawa folder kung saan naka save ang model
    model_dir.mkdir(exist_ok=True) # If exist or hindi
    
    model_path = model_dir / "iris_random_forest.joblib" # Exact file name ng model
    
    joblib.dump(model, model_path) # Pang execute para ilagay ang model sa may file name and path
    
    print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    main()