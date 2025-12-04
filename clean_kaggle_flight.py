import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib # To save the encoders for the App later

# 1. LOAD DATA
print("Loading Kaggle Data...")
try:
    df = pd.read_csv('flightdata.csv')
    # If the file uses "Unnamed: 0" index column, drop it
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
except FileNotFoundError:
    print("Error: 'flightdata.csv' not found. Please ensure the file is in the current directory.")
    exit(1)

print(f"Loaded {len(df)} rows.")

# 2. DROP USELESS COLUMNS
# 'flight' (e.g., SG-8709) is just an ID. It confuses the model.
df = df.drop('flight', axis=1)

# 3. ENCODE CATEGORICAL DATA (Text -> Numbers)
# We will save these encoders so we can use them in the App later!
encoders = {} 

categorical_cols = ['airline', 'source_city', 'departure_time', 'stops', 
                    'arrival_time', 'destination_city', 'class']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le # Save for later

# Save the dictionary of encoders
joblib.dump(encoders, 'flight_encoders.save')
print("Encoded columns: airline, city, time, stops, class.")

# 4. DEFINE X (Inputs) and y (Output)
# Inputs: Everything except Price
X = df.drop('price', axis=1).values
y = df['price'].values.reshape(-1, 1)

feature_names = df.drop('price', axis=1).columns.tolist()
print(f"Features used: {feature_names}")

# 5. SCALE DATA (Standardization)
scaler_x = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_x.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Save scalers
joblib.dump(scaler_x, 'flight_scaler_x.save')
joblib.dump(scaler_y, 'flight_scaler_y.save')

# 6. SPLIT & SAVE
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

np.savez('kaggle_flight_processed.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
print(" Ready! Data saved to 'kaggle_flight_processed.npz'")