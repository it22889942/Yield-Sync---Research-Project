# Yield-Sync---Research-Project

**Yield Sync – Smart Farm Assistant**
- Yield Sync is a mobile-based agriculture support application designed to assist farmers in making informed and timely farming decisions through digital technology. 
The application integrates multiple agriculture-related functionalities, including crop advisory, fertilizer recommendation, Equipment & Labour hiring, and market prediction, within a single platform.

**Main Functions**
- Intelligent Crop Advisory Tool
- Fertilizer reccomondation
- Equipment & Labour hiring
- Crop Price & Demand Prediction

---

# 1)  Intelligent Crop Advisory Tool
## 🔍Overview
The Crop Advisory feature in the Yield Sync mobile application provides farmers 
with practical guidance to select and manage crops effectively. It uses parameters such as 
PH, NPK, Temperature, Humadity rainfall conditions, and location to generate reliable recommendations.

---

## ✨Novelty
The Crop Advisory feature in Yield Sync uses a simple rule-based approach to recommend suitable crops 
based on soil conditions, weather, and seasonal factors. Unlike complex AI-driven systems, 
it is designed to be lightweight, accessible, and easy to use for farmers.

---

## ⚙️Key Features
- Weather and soil based crop reccomendation
- Soil parameter analysis
- Weather forecast






# 3) IoT ML Based Smart Fertilizer Recommendation

## 🔍 Overview

This module is part of the YieldSync – Smart Farm Assistant project.
It helps farmers select the correct fertilizer type, estimate the expected yield per acre, and calculate the exact fertilizer quantity required for their field using real-time soil data and machine learning.

The system is designed for Sri Lankan agriculture and supports crops such as rice, beetroot, radish, and red onion, considering soil conditions and crop growth stages to improve productivity and reduce fertilizer misuse.

---

## ✨ Novelty

The key novelty of this module is the integration of real-time IoT-based soil sensing with machine learning to provide crop-specific fertilizer recommendation, yield prediction, and fertilizer quantity calculation in a single automated workflow, which is not available in traditional fertilizer advisory methods.

---

## ⚙️ Key Features

- 🌱 Real-time soil data collection using IoT sensors
- 🧪 Soil pH and NPK-based fertilizer recommendation
- 🌾 Crop and growth-stage specific analysis
- 📈 Yield prediction per acre using ML models
- ⚖️ Fertilizer quantity calculation based on field size
- 📱 Simple, farmer-friendly mobile interface
  
---

## 🏗️ System Architecture – Fertilizer Recommendation
<pre>
Soil Data Collection (IoT Sensors)
  ↓
Data Processing & Validation
  ↓
ML Model (Fertilizer & Yield Prediction)
  ↓
Quantity Calculation Logic
  ↓
Farmer Recommendation Output
</pre>

---
## 🛠️ Technologies Used

- 🐍 Programming Language: Python
- 🤖 Machine Learning: scikit-learn
- 📂 Data Processing: Pandas, NumPy
- 📡 IoT Hardware: 7-in-1 Soil Sensor, ESP8266
- 🌐 Backend: Flask (API tested using Postman)
- 📱 UI Design: Figma
- 🔁 Version Control: Git and GitHub

---


# 4) Crop Price & Demand Prediction (SELL / HOLD Decision)

## 🔍Overview
This module is part of the **Yield Sync – Smart Farm Assistant** project.
It helps farmers decide the best time to sell their crops by predicting
future crop prices and market demand, and then providing a clear
**SELL or HOLD** recommendation.

The system is designed for Sri Lankan agriculture and considers
**Maha and Yala seasons**, **festival periods**, and **weather-related trends**
to reduce crop wastage and improve farmer income.

---

## ✨Novelty
The key novelty of this module is the **joint prediction of crop price and market demand**
and converting these predictions into a **simple, actionable SELL or HOLD decision**
tailored for Sri Lankan farmers.

---

## ⚙️Key Features
- 📈Future crop price prediction
- 📊Market demand forecasting (Rising / Stable / Falling)
- 🌱Season-aware modeling (Maha and Yala)
- 🎉Festival-based price adjustment
- ✅Clear SELL or HOLD recommendation
- 📝Short explanation for each decision

---

## 🏗️System Architecture – Crop Price & Demand Prediction
<pre>
Data Collection
   ↓
Data Processing
   ↓
ML Models (Price + Demand Prediction)
   ↓
Decision Logic (SELL / HOLD)
   ↓
Farmer Output
</pre>

---
## 🛠️Technologies Used
- 🐍Programming Language: Python  
- 🤖Machine Learning: scikit-learn  
- 📂Data Processing: Pandas, NumPy  
- 📉Visualization: Matplotlib  
- 🌐Backend: Flask  
- 🔁Version Control: Git and GitHub  

---
