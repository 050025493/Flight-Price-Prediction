import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib  # To load the encoders/scalers
import warnings

warnings.filterwarnings("ignore")

# 1. MODEL ARCHITECTURE
class FlightProNet(nn.Module):
    def __init__(self, input_dim):
        super(FlightProNet, self).__init__()
        self.layer1 = nn.Linear(input_dim, 256) 
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        self.layer2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        self.layer3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.2)
        self.output = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout1(self.relu(self.bn1(self.layer1(x))))
        x = self.dropout2(self.relu(self.bn2(self.layer2(x))))
        x = self.dropout3(self.relu(self.bn3(self.layer3(x))))
        x = self.output(x)
        return x

# 2. LOAD MODEL & SCALERS
print("Loading Flight Consultant...")

try:
    # Load Scalers & Encoders
    encoders = joblib.load('flight_encoders.save')
    scaler_x = joblib.load('flight_scaler_x.save')
    scaler_y = joblib.load('flight_scaler_y.save')
    
    # Load Model
    # We need to know input_dim. It was 9 features in your training.
    model = FlightProNet(input_dim=9) 
    model.load_state_dict(torch.load('best_flight_brain.pth'))
    model.eval()
    print(" System Online!")
    
except FileNotFoundError:
    print(" Error: Missing save files. Did you run 'clean_kaggle_flight.py' and training?")
    exit()

# 3. PREDICTION FUNCTION
def predict_flight_price(airline, source, dest, stops, ticket_class, days_left, departure_time):
    try:
        # 1. Encode Categorical Inputs
        # We use the saved LabelEncoders to convert "Vistara" -> 5
        a_code = encoders['airline'].transform([airline])[0]
        s_code = encoders['source_city'].transform([source])[0]
        d_code = encoders['destination_city'].transform([dest])[0]
        c_code = encoders['class'].transform([ticket_class])[0]
        
        # Departure Time (Morning/Evening...)
        dt_code = encoders['departure_time'].transform([departure_time])[0]
        
        # Stops ("zero", "one", "two_or_more") - Need to match dataset format exactly
        # The dataset used specific strings like "zero", "one", "two_or_more" 
        # We need to ensure we map user input to these strings first.
        stops_map = {0: "zero", 1: "one", 2: "two_or_more"}
        stops_str = stops_map.get(stops, "zero")
        st_code = encoders['stops'].transform([stops_str])[0]
        
        # Arrival Time? 
        # NOTE: Your training data included 'arrival_time'.
        # For simplicity, let's assume Arrival is 'Night' or same as Departure if short.
        # Ideally, we should ask the user, but let's default to 'Night' to keep it simple.
        arr_code = encoders['arrival_time'].transform(['Night'])[0]

        # Duration? 
        # We need to estimate duration based on cities.
        # Delhi -> Mumbai is usually ~2.17 hours (dataset average).
        duration = 2.17 

        # 2. Prepare Array [airline, source, departure, stops, arrival, dest, class, duration, days_left]
        # ORDER MATTERS! Check clean_kaggle_flight.py print(feature_names) to be sure.
        # Assuming order: ['airline', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class', 'duration', 'days_left']
        
        raw_input = np.array([[a_code, s_code, dt_code, st_code, arr_code, d_code, c_code, duration, days_left]])
        
        # 3. Scale
        input_scaled = scaler_x.transform(raw_input)
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
        
        # 4. Predict
        with torch.no_grad():
            pred_scaled = model(input_tensor)
            price = scaler_y.inverse_transform(pred_scaled.numpy())[0][0]
            
        return price
        
    except Exception as e:
        print(f"Error: {e}")
        return None

# 4. INTERACTIVE USER INPUT LOOP
print("\n--- FLIGHT PRICE PREDICTOR  ---")
print("Airlines: Vistara, Air_India, Indigo, GO_FIRST, AirAsia, SpiceJet")
print("Cities: Delhi, Mumbai, Bangalore, Kolkata, Hyderabad, Chennai")
print("Class: Economy, Business")
print("-" * 40)

while True:
    print("\nEnter Flight Details (or 'q' to quit):")
    
    airline = input("Airline: ").strip()
    if airline == 'q': break
    
    source = input("Source City: ").strip()
    dest = input("Destination City: ").strip()
    ticket_class = input("Class (Economy/Business): ").strip()
    days = int(input("Days Left (e.g., 1, 10, 50): "))
    
    # Optional inputs (Defaults)
    stops = 0 if ticket_class == "Business" else 1
    dep_time = "Morning"

    price = predict_flight_price(airline, source, dest, stops, ticket_class, days, dep_time)
    
    if price:
        print(f"\n Predicted Price: Rs. {price:,.2f}")
        if ticket_class == "Business" and price < 40000:
            print(" That's a cheap Business Class ticket!")
        elif ticket_class == "Economy" and price > 8000:
            print(" Prices are high right now.")