# Yield-Sync---Research-Project









# Crop Price & Demand Prediction Module (SELL / HOLD Decision)

## Project Overview
This module is part of the **Yield Sync – Smart Farm Assistant** project.
It helps farmers decide the best time to sell their crops by predicting
future crop prices and market demand, and then providing a clear
**SELL or HOLD** recommendation.

The system is designed for Sri Lankan agriculture and considers
**Maha and Yala seasons**, **festival periods**, and **weather-related trends**
to reduce crop wastage and improve farmer income.

---

## Novelty
The key novelty of this module is the **joint prediction of crop price and market demand**
and converting these predictions into a **simple, actionable SELL or HOLD decision**
tailored for Sri Lankan farmers.

---

## Key Features
- Future crop price prediction
- Market demand forecasting (Rising / Stable / Falling)
- Season-aware modeling (Maha and Yala)
- Festival-based price adjustment
- Clear SELL or HOLD recommendation
- Short explanation for each decision

---

## System Architecture – Crop Price & Demand Prediction Module

```mermaid
flowchart LR

A[Data Collection] --> B[Data Processing]
B --> C[Machine Learning Models]
C --> D[Decision Logic]
D --> E[Output to Farmer]

A --> A1[Crop Price Data]
A --> A2[Market Volume / Demand Data]
A --> A3[Seasonal Calendar<br/>(Maha / Yala)]
A --> A4[Festival Dates]
A --> A5[Weather Data]

B --> B1[Data Cleaning]
B --> B2[Feature Engineering]
B --> B3[Season & Festival Encoding]

C --> C1[Price Prediction Model]
C --> C2[Demand Forecasting Model]

D --> D1[SELL Recommendation]
D --> D2[HOLD Recommendation]

E --> E1[Predicted Future Price]
E --> E2[Demand Trend]
E --> E3[Reason for Decision]
  

---

## Technologies Used
- Programming Language: Python  
- Machine Learning: scikit-learn  
- Data Processing: Pandas, NumPy  
- Visualization: Matplotlib  
- Backend: Flask  
- Version Control: Git and GitHub  

---

## Folder Structure
