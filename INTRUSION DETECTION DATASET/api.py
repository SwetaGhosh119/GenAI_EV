from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import joblib
from typing import List
import uvicorn

app = FastAPI(title="IDS API", description="Intrusion Detection System API")

# Define model architectures (same as your training code)
class TransformerBinaryClassifier(nn.Module):
    def __init__(self, d_model):
        super(TransformerBinaryClassifier, self).__init__()
        self.input_layer = nn.Linear(d_model, 64)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=2, dim_feedforward=128, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.input_layer(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.fc(x)

class CNN_LSTM_IDS(nn.Module):
    def __init__(self, input_dim, conv_channels=32, lstm_hidden=64, lstm_layers=1, dropout=0.2):
        super(CNN_LSTM_IDS, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=conv_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(conv_channels)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(output_size=8)
        self.lstm = nn.LSTM(input_size=conv_channels, hidden_size=lstm_hidden, 
                           num_layers=lstm_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden*2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() == 3 and x.size(1) > 1:
            b, s, f = x.size()
            x = x.reshape(b, 1, s*f)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = out.mean(dim=1)
        out = self.fc(out)
        return out

# Load models at startup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transformer_model = TransformerBinaryClassifier(d_model=10).to(device)
transformer_model.load_state_dict(torch.load('best_model_augmented.pth', map_location=device))
transformer_model.eval()

cnn_lstm_model = CNN_LSTM_IDS(input_dim=10).to(device)
cnn_lstm_model.load_state_dict(torch.load('cnn_lstm_model.pth', map_location=device))
cnn_lstm_model.eval()

# Load scaler
scaler = joblib.load('scaler.pkl')

class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    confidence: float
    model_outputs: dict

@app.get("/")
def root():
    return {"message": "IDS API is running", "status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        # Validate input
        if len(request.features) != 10:
            raise HTTPException(status_code=400, detail="Expected 10 features")
        
        # Prepare input
        x = np.array(request.features).reshape(1, -1)
        x_scaled = scaler.transform(x)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(device)
        
        # Get predictions from both models
        with torch.no_grad():
            p1 = transformer_model(x_tensor).cpu().numpy().item()
            p2 = cnn_lstm_model(x_tensor).cpu().numpy().item()
        
        # Ensemble prediction (average)
        p_mean = (p1 + p2) / 2.0
        prediction = "Attack" if p_mean >= 0.5 else "Normal"
        confidence = abs(p_mean - 0.5) * 2  # Scale to 0-1
        
        return PredictionResponse(
            prediction=prediction,
            probability=float(p_mean),
            confidence=float(confidence),
            model_outputs={
                "transformer": float(p1),
                "cnn_lstm": float(p2)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
