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

## High-Level Architecture
1. Data Collection  
   - Crop price data (manually collected)  
   - Market volume and demand data (manually collected)  
   - Seasonal and festival calendar data  
   - Weather-related indicators  

2. Data Processing  
   - Data cleaning and normalization  
   - Feature engineering for season and festivals  

3. Machine Learning Models  
   - Price prediction model  
   - Demand forecasting model  
   - Decision logic for SELL / HOLD  

4. Output  
   - Predicted future price  
   - Demand trend  
   - Final recommendation with explanation  

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
