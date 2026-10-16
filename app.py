import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'Model'))

# Import feature parsers
from dataset_diabetes import engineer_features as eng_dia, GENETIC_COLS_DIA, LIFESTYLE_COLS_DIA, load_diabetes_data
from dataset_alzheimer import engineer_alzheimer_features as eng_alz, GENETIC_COLS_ALZ, LIFESTYLE_COLS_ALZ, load_alzheimer_data

app = Flask(__name__)
CORS(app)

print("\n⚙️ BOOTING UP AI PREDICTION SERVER...")
print("1. Loading and initializing Scalers from Base Datasets (this takes ~10 seconds)...")

# Globals to hold scalers and models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DIA_SCALER_GEN, DIA_SCALER_LIFE = None, None
ALZ_SCALER_GEN, ALZ_SCALER_LIFE = None, None
model_dia, model_alz = None, None
xgb_dia, xgb_alz = None, None
rf_dia = None

# ── MODEL ARCHITECTURES ──────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim * 2, dim), nn.Dropout(dropout)
        )
    def forward(self, x):
        return x + self.net(x)

class HIGANBinary(nn.Module):
    def __init__(self, n_genetic, n_lifestyle, embed_dim=256, n_heads=8, dropout=0.3):
        super().__init__()
        self.gen_attn = nn.Sequential(nn.Linear(n_genetic, n_genetic * 2), nn.GELU(), nn.Linear(n_genetic * 2, n_genetic), nn.Sigmoid())
        self.life_attn = nn.Sequential(nn.Linear(n_lifestyle, n_lifestyle * 2), nn.GELU(), nn.Linear(n_lifestyle * 2, n_lifestyle), nn.Sigmoid())
        
        self.gen_enc = nn.Sequential(nn.Linear(n_genetic, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout), nn.Linear(512, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
        self.gen_res = nn.Sequential(ResBlock(embed_dim, dropout), ResBlock(embed_dim, dropout))
        
        self.life_enc = nn.Sequential(nn.Linear(n_lifestyle, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout), nn.Linear(512, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
        self.life_res = nn.Sequential(ResBlock(embed_dim, dropout), ResBlock(embed_dim, dropout))

        self.gen_to_life = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.life_to_gen = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm_g = nn.LayerNorm(embed_dim)
        self.norm_l = nn.LayerNorm(embed_dim)

        self.gate = nn.Sequential(nn.Linear(embed_dim*2, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim*2), nn.Sigmoid())
        self.proj = nn.Sequential(nn.Linear(embed_dim*2, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
        self.refine = nn.Sequential(ResBlock(embed_dim, dropout), ResBlock(embed_dim, dropout))

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
    def forward(self, x_gen, x_life):
        g = self.gen_res(self.gen_enc(x_gen * self.gen_attn(x_gen)))
        l = self.life_res(self.life_enc(x_life * self.life_attn(x_life)))
        g_ctx, _ = self.gen_to_life(g.unsqueeze(1), l.unsqueeze(1), l.unsqueeze(1))
        l_ctx, _ = self.life_to_gen(l.unsqueeze(1), g.unsqueeze(1), g.unsqueeze(1))
        g_ctx = self.norm_g(g + g_ctx.squeeze(1))
        l_ctx = self.norm_l(l + l_ctx.squeeze(1))
        fused = torch.cat([g_ctx, l_ctx], dim=-1)
        fused = self.refine(self.proj(fused * self.gate(fused)))
        return self.classifier(fused), fused

# Small version for Alzheimer's
class HIGANBinaryAlz(nn.Module):
    def __init__(self, n_genetic, n_lifestyle, embed_dim=128, n_heads=4, dropout=0.25):
        super().__init__()
        self.gen_attn = nn.Sequential(nn.Linear(n_genetic, n_genetic * 2), nn.GELU(), nn.Linear(n_genetic * 2, n_genetic), nn.Sigmoid())
        self.life_attn = nn.Sequential(nn.Linear(n_lifestyle, n_lifestyle * 2), nn.GELU(), nn.Linear(n_lifestyle * 2, n_lifestyle), nn.Sigmoid())
        
        self.gen_enc = nn.Sequential(nn.Linear(n_genetic, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
        self.gen_res = nn.Sequential(ResBlock(embed_dim, dropout), ResBlock(embed_dim, dropout))
        
        self.life_enc = nn.Sequential(nn.Linear(n_lifestyle, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
        self.life_res = nn.Sequential(ResBlock(embed_dim, dropout), ResBlock(embed_dim, dropout))

        self.gen_to_life = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.life_to_gen = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm_g = nn.LayerNorm(embed_dim)
        self.norm_l = nn.LayerNorm(embed_dim)

        self.gate = nn.Sequential(nn.Linear(embed_dim*2, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim*2), nn.Sigmoid())
        self.proj = nn.Sequential(nn.Linear(embed_dim*2, embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
        self.refine = nn.Sequential(ResBlock(embed_dim, dropout), ResBlock(embed_dim, dropout))

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(32, 2)
        )
    def forward(self, x_gen, x_life):
        g = self.gen_res(self.gen_enc(x_gen * self.gen_attn(x_gen)))
        l = self.life_res(self.life_enc(x_life * self.life_attn(x_life)))
        g_ctx, _ = self.gen_to_life(g.unsqueeze(1), l.unsqueeze(1), l.unsqueeze(1))
        l_ctx, _ = self.life_to_gen(l.unsqueeze(1), g.unsqueeze(1), g.unsqueeze(1))
        g_ctx = self.norm_g(g + g_ctx.squeeze(1))
        l_ctx = self.norm_l(l + l_ctx.squeeze(1))
        fused = torch.cat([g_ctx, l_ctx], dim=-1)
        fused = self.refine(self.proj(fused * self.gate(fused)))
        return self.classifier(fused), fused

# ── INITIALIZATION ───────────────────────────────────────────────
with app.app_context():
    # 1. Diabetes Configuration
    print("Fetching Diabetes Configuration...")
    _, _, _, DIA_SCALER_GEN, DIA_SCALER_LIFE, n_gen_d, n_life_d = load_diabetes_data()
    model_dia = HIGANBinary(n_gen_d, n_life_d, embed_dim=256).to(device)
    model_dia.load_state_dict(torch.load(os.path.join(BASE_DIR, 'Model', 'saved_models', 'higan_diabetes_best.pt'), map_location=device))
    model_dia.eval()
    xgb_dia = joblib.load(os.path.join(BASE_DIR, 'Model', 'saved_models', 'xgb_diabetes.pkl'))
    rf_dia = joblib.load(os.path.join(BASE_DIR, 'Model', 'saved_models', 'rf_diabetes.pkl'))

    # 2. Alzheimer Configuration
    print("Fetching Alzheimer's Configuration...")
    _, _, _, ALZ_SCALER_GEN, ALZ_SCALER_LIFE, n_gen_a, n_life_a = load_alzheimer_data()
    model_alz = HIGANBinaryAlz(n_gen_a, n_life_a, embed_dim=128).to(device)
    model_alz.load_state_dict(torch.load(os.path.join(BASE_DIR, 'Model', 'saved_models', 'higan_alzheimer_best.pt'), map_location=device))
    model_alz.eval()
    xgb_alz = joblib.load(os.path.join(BASE_DIR, 'Model', 'saved_models', 'xgb_alzheimer.pkl'))
    print("✅ SERVER READY.")

@app.route('/')
def home():
    return render_template('index.html')

# ── INPUT CLAMPING (Enforce valid BRFSS / Dataset ranges) ─────────
CLAMP_RANGES_DIA = {
    'Age': (1, 13), 'Sex': (0, 1), 'BMI': (10, 60),
    'HighBP': (0, 1), 'HighChol': (0, 1), 'CholCheck': (0, 1),
    'Smoker': (0, 1), 'Stroke': (0, 1), 'HeartDiseaseorAttack': (0, 1),
    'PhysActivity': (0, 1), 'Fruits': (0, 1), 'Veggies': (0, 1),
    'HvyAlcoholConsump': (0, 1), 'AnyHealthcare': (0, 1), 'NoDocbcCost': (0, 1),
    'DiffWalk': (0, 1),
    'GenHlth': (1, 5), 'MentHlth': (0, 30), 'PhysHlth': (0, 30),
    'Education': (1, 6), 'Income': (1, 8)
}

CLAMP_RANGES_ALZ = {
    'Age': (60, 90), 'Gender': (0, 1), 'Ethnicity': (0, 3), 'EducationLevel': (0, 3),
    'BMI': (15, 40), 'Smoking': (0, 1), 'AlcoholConsumption': (0, 20),
    'PhysicalActivity': (0, 10), 'DietQuality': (0, 10), 'SleepQuality': (4, 10),
    'FamilyHistoryAlzheimers': (0, 1), 'CardiovascularDisease': (0, 1),
    'Diabetes': (0, 1), 'Depression': (0, 1), 'HeadInjury': (0, 1), 'Hypertension': (0, 1),
    'SystolicBP': (90, 180), 'DiastolicBP': (60, 120),
    'CholesterolTotal': (150, 300), 'CholesterolLDL': (50, 200),
    'CholesterolHDL': (20, 100), 'CholesterolTriglycerides': (50, 400),
    'MMSE': (0, 30), 'FunctionalAssessment': (0, 10),
    'MemoryComplaints': (0, 1), 'BehavioralProblems': (0, 1),
    'ADL': (0, 10), 'Confusion': (0, 1), 'Disorientation': (0, 1),
    'PersonalityChanges': (0, 1), 'DifficultyCompletingTasks': (0, 1), 'Forgetfulness': (0, 1)
}

def clamp_inputs(df, ranges):
    """Clamp all DataFrame columns to their valid dataset ranges."""
    for col, (lo, hi) in ranges.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)
    return df

@app.route('/api/predict/diabetes', methods=['POST'])
def predict_diabetes():
    try:
        data = request.json
        # Convert dictionary to DataFrame for feature engineering
        df = pd.DataFrame([data])
        
        # Ensure correct datatypes
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # CLAMP inputs to valid BRFSS ranges
        df = clamp_inputs(df, CLAMP_RANGES_DIA)

        # 1. Apply feature engineering
        df_eng = eng_dia(df)

        gen_cols  = [c for c in GENETIC_COLS_DIA if c in df_eng.columns]
        life_cols = [c for c in LIFESTYLE_COLS_DIA if c in df_eng.columns]

        X_gen = df_eng[gen_cols].values.astype(np.float32)
        X_life = df_eng[life_cols].values.astype(np.float32)

        # 2. Scale
        X_gen_sc = DIA_SCALER_GEN.transform(X_gen)
        X_life_sc = DIA_SCALER_LIFE.transform(X_life)

        # 3. Model Inference (Retrieve all AI brains)
        model_dia.eval()
        with torch.no_grad():
            t_gen = torch.tensor(X_gen_sc).to(device)
            t_life = torch.tensor(X_life_sc).to(device)
            # HIGAN Deep Learning Prediction
            logits, feats = model_dia(t_gen, t_life)
            probs = torch.softmax(logits, dim=-1)
            higan_prob = float(probs[:, 1].item())
            
            # XGBoost Prediction (using HIGAN extracted graph embeddings)
            xgb_prob = float(xgb_dia.predict_proba(feats.cpu().numpy())[:, 1][0])

        # 4. Standard Tabular Prediction (High-Fidelity Random Forest)
        raw_feat = np.hstack([X_gen_sc, X_life_sc])
        rf_prob = float(rf_dia.predict_proba(raw_feat)[:, 1][0])

        # 5. Hybrid Triple-Ensemble Weightage 
        # (Heavily weighted towards Random Forest to strictly maintain the >90% empirical accuracy)
        final_prob = (0.20 * higan_prob) + (0.20 * xgb_prob) + (0.60 * rf_prob)
        prediction = bool(final_prob >= 0.5)

        return jsonify({
            'success': True,
            'prediction': prediction,
            'risk_percentage': round(float(final_prob) * 100, 2),
            'higan_confidence': round(higan_prob * 100, 2),
            'xgb_confidence': round(xgb_prob * 100, 2),
            'rf_confidence': round(rf_prob * 100, 2)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/predict/alzheimer', methods=['POST'])
def predict_alzheimer():
    try:
        data = request.json
        df = pd.DataFrame([data])
        
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # CLAMP inputs to valid dataset ranges
        df = clamp_inputs(df, CLAMP_RANGES_ALZ)

        df_eng = eng_alz(df)

        gen_cols  = [c for c in GENETIC_COLS_ALZ if c in df_eng.columns]
        life_cols = [c for c in LIFESTYLE_COLS_ALZ if c in df_eng.columns]

        X_gen = df_eng[gen_cols].values.astype(np.float32)
        X_life = df_eng[life_cols].values.astype(np.float32)

        X_gen_sc = ALZ_SCALER_GEN.transform(X_gen)
        X_life_sc = ALZ_SCALER_LIFE.transform(X_life)

        model_alz.eval()
        with torch.no_grad():
            t_gen = torch.tensor(X_gen_sc).to(device)
            t_life = torch.tensor(X_life_sc).to(device)
            logits, feats = model_alz(t_gen, t_life)
            probs = torch.softmax(logits, dim=-1)
            higan_prob = float(probs[:, 1].item())
            
            xgb_prob = float(xgb_alz.predict_proba(feats.cpu().numpy())[:, 1][0])

        final_prob = (0.4 * higan_prob) + (0.6 * xgb_prob)
        prediction = bool(final_prob >= 0.5)

        return jsonify({
            'success': True,
            'prediction': prediction,
            'risk_percentage': round(float(final_prob) * 100, 2),
            'higan_confidence': round(higan_prob * 100, 2),
            'xgb_confidence': round(xgb_prob * 100, 2)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
