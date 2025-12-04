# Data Directory

## Overview

This directory contains all datasets used in the Yield-Sync crop price prediction and demand forecasting project.

---

## Dataset Files

### 1. 2020_2024.csv

The primary dataset containing daily agricultural market data.

| Column    | Description                                    | Type     |
| --------- | ---------------------------------------------- | -------- |
| Date      | Transaction date                               | datetime |
| market    | Market name (15 markets)                       | string   |
| item      | Crop type (Rice, Beetroot, Raddish, Red Onion) | string   |
| price     | Daily price in LKR                             | float    |
| volume_MT | Trading volume in Metric Tons                  | float    |

- **Records**: 109,620 rows
- **Period**: January 1, 2020 to December 31, 2024 (5 years)
- **Granularity**: Daily per market per crop

---

### 2. weather_data.csv

Daily weather observations for the region.

| Column            | Description         | Unit        |
| ----------------- | ------------------- | ----------- |
| Date              | Observation date    | datetime    |
| temperature_avg_C | Average temperature | Celsius     |
| temperature_min_C | Minimum temperature | Celsius     |
| temperature_max_C | Maximum temperature | Celsius     |
| rainfall_mm       | Rainfall amount     | millimeters |
| humidity_percent  | Relative humidity   | percentage  |
| wind_speed_kmh    | Wind speed          | km/hour     |

- **Records**: 1,827 rows (5 years of daily data)

---

### 3. seasonal_indicators.csv

Cultivation season and harvest period indicators.

| Column                | Description               | Values                                           |
| --------------------- | ------------------------- | ------------------------------------------------ |
| Date                  | Date                      | datetime                                         |
| season                | Climate season            | Maha (Oct-Mar) / Yala (Apr-Sep)                  |
| cultivation_season    | Active cultivation period | Maha_Cultivation / Yala_Cultivation / Off_Season |
| harvest_period_rice   | Rice harvest indicator    | 0 or 1                                           |
| vegetable_peak_season | Vegetable peak season     | 0 or 1                                           |

- **Records**: 1,827 rows

---

### 4. festival_holidays.csv

Sri Lankan public holidays and festival periods.

| Column                | Description                     | Values     |
| --------------------- | ------------------------------- | ---------- |
| Date                  | Date                            | datetime   |
| is_public_holiday     | Public holiday flag             | 0 or 1     |
| is_day_before_holiday | Day before holiday              | 0 or 1     |
| near_major_holiday    | Within 3 days of major holiday  | 0 or 1     |
| demand_multiplier     | Expected demand increase factor | 1.0 to 1.5 |

- **Records**: 1,827 rows
- **Major Holidays**: Sinhala/Tamil New Year, Vesak, Poson, Christmas, Deepavali, Eid

---

### 5. model_ready_data.csv

Preprocessed dataset with all features merged, ready data for model training.

| Feature Category | Columns                                                                                                |
| ---------------- | ------------------------------------------------------------------------------------------------------ |
| Identifiers      | Date, market, item                                                                                     |
| Price/Volume     | price, volume_MT                                                                                       |
| Weather          | temperature_avg_C, temperature_min_C, temperature_max_C, rainfall_mm, humidity_percent, wind_speed_kmh |
| Seasonal         | season, cultivation_season, harvest_period_rice, vegetable_peak_season                                 |
| Holidays         | is_public_holiday, is_day_before_holiday, near_major_holiday, demand_multiplier                        |
| Lag Features     | price_lag_1, price_lag_7, price_rolling_7_mean, volume_lag_1                                           |

- **Records**: 109,200 rows
- **Columns**: 23
