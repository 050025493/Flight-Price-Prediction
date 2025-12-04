from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from datetime import datetime


app = FastAPI()

# Allow the HTML file to talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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


print(" Starting Engine...")
try:
    encoders = joblib.load('flight_encoders.save')
    scaler_x = joblib.load('flight_scaler_x.save')
    scaler_y = joblib.load('flight_scaler_y.save')
    model = FlightProNet(input_dim=9)
    model.load_state_dict(torch.load('best_flight_brain.pth'))
    model.eval()
    
    # Load Schedule
    schedule_df = pd.read_csv('flightdata.csv')
    cols = ['airline', 'source_city', 'destination_city', 'class', 'departure_time', 'arrival_time', 'duration', 'stops']
    schedule_df = schedule_df[cols].drop_duplicates()
    
    print(" System Online!")
except Exception as e:
    print(f" Error loading files: {e}")



# A. Get Dropdown Options
@app.get("/options")
def get_options():
    return {
        "cities": list(encoders['source_city'].classes_),
        "airlines": ["All"] + list(encoders['airline'].classes_)
    }

# B. Predict Price
class SearchRequest(BaseModel):
    source: str
    dest: str
    date: str
    cls: str
    airline: str

@app.post("/search")
def search_flights(req: SearchRequest):
    # 1. Calc Days Left
    travel_date = datetime.strptime(req.date, "%Y-%m-%d")
    days_left = (travel_date - datetime.today()).days + 1
    
    # 2. Filter Database
    matches = schedule_df[
        (schedule_df['source_city'] == req.source) &
        (schedule_df['destination_city'] == req.dest) &
        (schedule_df['class'] == req.cls)
    ]
    
    if req.airline != "All":
        matches = matches[matches['airline'] == req.airline]
        
    if len(matches) == 0:
        return {"status": "empty", "message": "No flights found."}
    
    # 3. AI Prediction
    inputs = []
    flights = []
    
    for _, row in matches.iterrows():
        # Encode
        a = encoders['airline'].transform([row['airline']])[0]
        s = encoders['source_city'].transform([row['source_city']])[0]
        d = encoders['destination_city'].transform([row['destination_city']])[0]
        c = encoders['class'].transform([row['class']])[0]
        dt = encoders['departure_time'].transform([row['departure_time']])[0]
        at = encoders['arrival_time'].transform([row['arrival_time']])[0]
        st_val = encoders['stops'].transform([row['stops']])[0]
        
        inputs.append([a, s, dt, st_val, at, d, c, row['duration'], days_left])
        
        # Keep flight details for response
        flights.append({
            "airline": row['airline'],
            "depart": row['departure_time'],
            "arrive": row['arrival_time'],
            "duration": row['duration'],
            "stops": row['stops']
        })

    # Batch Predict
    input_tensor = torch.tensor(scaler_x.transform(np.array(inputs)), dtype=torch.float32)
    with torch.no_grad():
        prices = scaler_y.inverse_transform(model(input_tensor).numpy())
    
    # Attach prices to flights
    for i, f in enumerate(flights):
        f['price'] = int(prices[i][0])
        
    # 4. Recommendation Logic
    # Convert to DataFrame for easy sorting
    res_df = pd.DataFrame(flights)
    min_p = res_df['price'].min()
    min_d = res_df['duration'].min()
    res_df['pain'] = (res_df['price'] - min_p) + ((res_df['duration'] - min_d) * 500)
    
    best = res_df.sort_values('pain').iloc[0].to_dict()
    cheapest = res_df.sort_values('price').iloc[0].to_dict()
    
    # Convert back to list for JSON response
    all_flights = res_df.sort_values('price').to_dict(orient='records')
    
    return {
        "status": "success",
        "best": best,
        "cheapest": cheapest,
        "all": all_flights,
        "days_left": days_left
    }

# Run with: uvicorn server:app --reload