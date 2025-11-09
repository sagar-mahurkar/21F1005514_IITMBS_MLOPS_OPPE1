import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train_model(data_path="./data.csv", model_dir="artifacts"):
    # --- Load data ---
    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    print(f"✅ Loaded data: {df.shape}")

    # --- Feature & target selection ---
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'rolling_avg_10', 'volume_sum_10']
    target_col = 'target'

    # Preserve timestamp (not used in training)
    timestamps = df['timestamp']

    X = df[feature_cols]
    y = df[target_col]

    # --- Train/test split (chronological) ---
    X_train, X_test, y_train, y_test, ts_train, ts_test = train_test_split(
        X, y, timestamps, test_size=0.2, shuffle=False
    )

    # --- Train model ---
    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    # --- Evaluate ---
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Model accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # --- Save model ---
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, model_path)

    print(f"\n💾 Model saved to: {model_path}")

    return model


if __name__ == "__main__":
    train_model()
