require('dotenv').config();
const express = require('express');
const cors = require('cors');

const authRoutes = require('./routes/authRoutes');
const patientRoutes = require('./routes/patientRoutes');

const app = express();
app.use(cors());
app.use(express.json());

// API Map
app.use('/api/auth', authRoutes);
app.use('/api/patient', patientRoutes);

app.get('/api/health', (req, res) => {
    res.json({ message: 'Patient Management API MVC Backend is running securely.' });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`🚀 NodeJS MVC Backend securely running on port ${PORT}`);
});
