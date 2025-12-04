import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

#define model architecture
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

# load everything
print("Loading Smart Travel Agent...")

try:
    encoders = joblib.load('flight_encoders.save')
    scaler_x = joblib.load('flight_scaler_x.save')
    scaler_y = joblib.load('flight_scaler_y.save')
    model = FlightProNet(input_dim=9) 
    model.load_state_dict(torch.load('best_flight_brain.pth'))
    model.eval()
    
    # Load Schedule
    schedule_df = pd.read_csv('flightdata.csv')
    schedule_df = schedule_df[['airline', 'source_city', 'destination_city', 'class', 'departure_time', 'arrival_time', 'duration', 'stops']].drop_duplicates()
    
    print(" Agent Ready!")
    
except FileNotFoundError:
    print(" Error: Files missing. Run training first.")
    exit()

#helpfer functions
def clean_input(user_input, category_name):
    valid_options = encoders[category_name].classes_
    if user_input in valid_options: return user_input
    for option in valid_options:
        if user_input.lower().strip() == option.lower(): return option
    if category_name == 'airline':
        if "air" in user_input.lower() and "india" in user_input.lower(): return "Air_India"
        if "go" in user_input.lower() and "first" in user_input.lower(): return "GO_FIRST"
    return None

def predict_batch(rows, days_left):
    inputs = []
    for _, row in rows.iterrows():
        a_code = encoders['airline'].transform([row['airline']])[0]
        s_code = encoders['source_city'].transform([row['source_city']])[0]
        d_code = encoders['destination_city'].transform([row['destination_city']])[0]
        c_code = encoders['class'].transform([row['class']])[0]
        dt_code = encoders['departure_time'].transform([row['departure_time']])[0]
        at_code = encoders['arrival_time'].transform([row['arrival_time']])[0]
        stops_str = row['stops']
        st_code = encoders['stops'].transform([stops_str])[0]
        duration = row['duration']
        
        inputs.append([a_code, s_code, dt_code, st_code, at_code, d_code, c_code, duration, days_left])
    
    input_scaled = scaler_x.transform(np.array(inputs))
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
    
    with torch.no_grad():
        preds_scaled = model(input_tensor)
        prices = scaler_y.inverse_transform(preds_scaled.numpy())
    
    return prices

def get_recommendation(df):
    """
    Finds the BEST flight by balancing Price vs. Duration.
    Rule: We are willing to pay Rs. 500 extra to save 1 hour.
    """
    df = df.copy()
    min_price = df['predicted_price'].min()
    min_duration = df['duration'].min()
    
    # Pain Score: Lower is better
    df['pain_score'] = (df['predicted_price'] - min_price) + ((df['duration'] - min_duration) * 500)
    df = df.sort_values('pain_score')
    
    return df.iloc[0], df.sort_values('predicted_price').iloc[0]

#interactive loop
print("\n---  SMART FLIGHT AGENT (DATE ENABLED)  ---")

while True:
    print("\n" + "-"*40)
    print("Where do you want to fly? (or 'q')")
    
    source_in = input("From (e.g. Delhi): ").strip()
    if source_in == 'q': break
    dest_in = input("To (e.g. Mumbai): ").strip()
    
    # --- NEW: DATE HANDLING ---
    while True:
        date_str = input("Travel Date (YYYY-MM-DD): ").strip()
        try:
            travel_date = datetime.strptime(date_str, "%Y-%m-%d")
            today = datetime.now()
            
            # Calculate Days Left
            delta = travel_date - today
            days_left = delta.days + 1 # +1 because if it's tomorrow, delta is 0
            
            if days_left < 1:
                print(" You cannot book for the past! Try a future date.")
            elif days_left > 50:
                # Kaggle dataset usually caps around 50 days, so we warn user
                print(f" Note: {days_left} days is far out. Predictions might be generic.")
                break
            else:
                break
        except ValueError:
            print(" Invalid format. Use YYYY-MM-DD (e.g., 2025-12-25)")

    class_in = input("Class (e/b): ").lower()
    ticket_class = "Business" if class_in.startswith('b') else "Economy"
    
    print(f"\n🔎 Searching for flights on {travel_date.strftime('%B %d, %Y')} ({days_left} days left)...")
    
    # CLEAN INPUTS
    source = clean_input(source_in, 'source_city')
    dest = clean_input(dest_in, 'destination_city')
    
    if None in [source, dest]:
        print(" City not found. Try Delhi, Mumbai, Bangalore, Kolkata, Hyderabad, Chennai.")
        continue

    # FIND FLIGHTS
    matches = schedule_df[
        (schedule_df['source_city'] == source) &
        (schedule_df['destination_city'] == dest) &
        (schedule_df['class'] == ticket_class)
    ]
    
    if len(matches) == 0:
        print(" No flights found for this route.")
        continue
        
    # PREDICT PRICES
    prices = predict_batch(matches, days_left)
    matches['predicted_price'] = prices
    
    # RECOMMENDATION
    best_flight, cheapest_flight = get_recommendation(matches)
    
    # DISPLAY
    print(f"\n Found {len(matches)} flights. Here are the top options:\n")
    print(f"{'AIRLINE':<12} {'DEPART':<10} {'DURATION':<10} {'STOPS':<10} {'PRICE':<10} {'NOTES'}")
    print("="*75)
    
    sorted_matches = matches.sort_values('predicted_price').head(5)
    
    for _, row in sorted_matches.iterrows():
        note = ""
        if row.equals(best_flight): note = "⭐ BEST VALUE"
        elif row.equals(cheapest_flight): note = "💰 CHEAPEST"
            
        print(f"{row['airline']:<12} {row['departure_time']:<10} {row['duration']:<10} {row['stops']:<10} {int(row['predicted_price']):<10} {note}")

    print("="*75)
    print("\n AI RECOMMENDATION:")
    if best_flight.equals(cheapest_flight):
        print(f"Go with {best_flight['airline']} ({best_flight['departure_time']}) at Rs. {int(best_flight['predicted_price'])}.")
        print("It is the cheapest AND the best value option.")
    else:
        diff = int(best_flight['predicted_price'] - cheapest_flight['predicted_price'])
        time_diff = cheapest_flight['duration'] - best_flight['duration']
        print(f"I recommend {best_flight['airline']} ({best_flight['departure_time']}) for Rs. {int(best_flight['predicted_price'])}.")
        print(f"Why? It costs Rs. {diff} more than the cheapest option, but saves you {time_diff:.1f} hours.")