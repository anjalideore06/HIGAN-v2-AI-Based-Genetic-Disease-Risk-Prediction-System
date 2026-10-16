# AI-Based Genetic Disorder Risk Prediction

## Overview
This project is an AI-powered Clinical Decision Support System that predicts the risk of **Alzheimer's Disease** and **Type 2 Diabetes** using a committee-based machine learning architecture. It combines **HIGAN (Deep Learning)**, **XGBoost**, and **Random Forest** to provide reliable clinical risk predictions.

## Project Evolution
- Initially started as a Diabetes prediction system.
- Expanded into a dual-disease prediction platform.
- Enhanced with a committee model to improve reliability and reduce false predictions.

## Tech Stack

### Frontend
- React.js
- Vite
- Tailwind CSS

### Backend
- Node.js
- Express.js

### Database
- PostgreSQL

## Key Features
- Dual prediction modules for Alzheimer's Disease and Type 2 Diabetes.
- Secure user authentication.
- Patient history and health timeline management.
- AI-powered risk prediction.
- Modern responsive UI with premium dark theme.

## Dataset
### Diabetes Dataset
- 253,680 records.
- Lifestyle, demographic, and health-related features.
- Includes engineered features such as Metabolic Syndrome Index and Socioeconomic Score.

### Alzheimer's Dataset
- Clinical neuroimaging and healthcare data.
- Cognitive assessment features including MMSE and other medical indicators.

## Data Preprocessing
- Data cleaning and feature engineering.
- Class imbalance handled using **SMOTE + Tomek Links**.
- Oversampling applied only on training data to prevent data leakage.

## AI Models

### Base Models
- HIGAN (Hierarchical Graph Attention Network)
- XGBoost
- Random Forest

### Committee Model
#### Diabetes
```
Final Probability =
0.20 × HIGAN +
0.20 × XGBoost +
0.60 × Random Forest
```

#### Alzheimer's
```
Final Probability =
0.40 × HIGAN +
0.60 × XGBoost
```

## Performance
| Disease | Accuracy | ROC-AUC |
|---------|----------|---------|
| Alzheimer's | 93.81% | 0.9582 |
| Diabetes | High | 0.9245 |

The committee approach improves robustness, increases AUC, and minimizes the risk of incorrect clinical decisions.

## Future Scope
- DNA sequencing (VCF) integration.
- Wearable device integration.
- Real-time health monitoring.
- Explainable AI.
- Cloud deployment.
- Multi-disease prediction.

## Why Committee Learning?
- Better prediction stability.
- No single point of failure.
- Improved sensitivity for early disease detection.
- Higher clinical reliability compared to individual models.

## Authors
Project developed as an AI-based Clinical Decision Support System for Genetic Disorder Risk Prediction.

## License
For academic and research purposes.



## 🚀 How to Run the Project

### Step 1: Open PowerShell and Allow Script Execution

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Step 2: Activate the Python Virtual Environment

**Option 1: Using the full path**

```powershell
&C:\Users\anjal\OneDrive\Desktop\Tanu_Dii_AI_Based_Genetic_Disorder_Risk_Prediction\venv\Scripts\Activate.ps1
```

**OR**

**Option 2: From the project directory**

```powershell
venv\Scripts\Activate.ps1
```

### Step 3: Start the Python Backend

```bash
python app.py
```

### Step 4: Open a Second Terminal and Start the Node.js Backend

```bash
cd NodeJS_Backend
npm run dev
```

### Step 5: Open a Third Terminal and Start the React Frontend

```bash
cd React_Frontend
npm run dev
```

## 📌 Running Summary

| Terminal | Command |
|----------|---------|
| Terminal 1 | Activate the virtual environment → `python app.py` |
| Terminal 2 | `cd NodeJS_Backend` → `npm run dev` |
| Terminal 3 | `cd React_Frontend` → `npm run dev` |

After all three services are running, open the React application in your browser (typically **http://localhost:5173**).
