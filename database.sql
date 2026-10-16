-- =========================================================================
-- HIGAN NEXUS: AI PATIENT MANAGEMENT SYSTEM
-- PostgreSQL Schema
-- Run this script sequentially in PgAdmin4 query tool using database: fyp_health
-- =========================================================================

-- 1. Users Table (Role Based Access Control)
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    role VARCHAR(20) NOT NULL CHECK (role IN ('PATIENT', 'DOCTOR')),
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Patient Profiles (Linked to Users if role = PATIENT)
CREATE TABLE IF NOT EXISTS patients (
    id SERIAL PRIMARY KEY,
    user_id INT UNIQUE REFERENCES users(id) ON DELETE CASCADE,
    doctor_id INT REFERENCES users(id) ON DELETE SET NULL, -- Assigned doctor
    age INT,
    gender VARCHAR(20),
    blood_group VARCHAR(10),
    contact_number VARCHAR(20),
    address TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Diagnostics & Risk Scores (ML Model Outputs)
CREATE TABLE IF NOT EXISTS diagnostics (
    id SERIAL PRIMARY KEY,
    patient_id INT REFERENCES patients(id) ON DELETE CASCADE,
    doctor_id INT REFERENCES users(id) ON DELETE SET NULL, -- Doctor who triggered or reviewed it
    model_type VARCHAR(50) NOT NULL CHECK (model_type IN ('diabetes', 'alzheimer')),
    clinical_data JSONB NOT NULL, -- Raw attributes passed to ML (e.g., BMI, HighBP, etc.)
    risk_percentage DECIMAL(5,2) NOT NULL, -- The XGBoost + HIGAN Ensemble score
    final_prediction BOOLEAN NOT NULL,
    doctor_annotation TEXT, -- Doctor's notes on the trend
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4. Health Timeline & Events
CREATE TABLE IF NOT EXISTS health_timeline (
    id SERIAL PRIMARY KEY,
    patient_id INT REFERENCES patients(id) ON DELETE CASCADE,
    event_type VARCHAR(100) NOT NULL, -- e.g., 'Symptom Updated', 'Doctor Visit', 'ML Scan'
    description TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert Mock Doctor (Password: password123)
INSERT INTO users (role, first_name, last_name, email, password_hash) 
VALUES ('DOCTOR', 'Dr. Sarah', 'Connor', 'doctor@higan.com', '$2y$10$gMpd3/tT5r/kK8w/9y8DheGjR//N2Y.tX6JcO0oB68m6.z.sL81R2')
ON CONFLICT (email) DO NOTHING;

-- Insert Mock Patient (Password: password123)
INSERT INTO users (role, first_name, last_name, email, password_hash) 
VALUES ('PATIENT', 'John', 'Doe', 'john@patient.com', '$2y$10$gMpd3/tT5r/kK8w/9y8DheGjR//N2Y.tX6JcO0oB68m6.z.sL81R2')
ON CONFLICT (email) DO NOTHING;

INSERT INTO patients (user_id, doctor_id, age, gender) 
VALUES (2, 1, 45, 'Male')
ON CONFLICT (user_id) DO NOTHING;
