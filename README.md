# PulseRetain — SaaS Attrition Intelligence 📡

> **An end-to-end Data Analytics & Machine Learning portfolio project demonstrating full-stack data engineering, predictive modeling, and interactive visualization.**

![PulseRetain Dashboard UI](https://img.shields.io/badge/UI-Spark_Admin_Theme-B4F105?style=for-the-badge&logoColor=051C12)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-Data_Warehouse-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Machine_Learning-Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

---

## 📌 Project Overview

**PulseRetain** is a comprehensive analytics platform built to solve one of the most critical problems in SaaS: **Customer Churn**. 

This project simulates a real-world data workflow. It generates raw subscription and engagement data, processes it through a robust ETL pipeline, stores the clean data in an SQLite Data Warehouse, trains multiple Machine Learning models to predict churn risk, and visualizes the insights through a premium, interactive Streamlit dashboard.

### 🎯 Key Objectives Achieved:
- **Data Engineering:** Automated synthetic data generation, rigorous data validation, and feature engineering.
- **SQL Data Warehousing:** Designed a relational database schema in SQLite with pre-computed KPI views for downstream BI tools (Power BI/Tableau).
- **Predictive Analytics:** Trained and evaluated multiple models (Logistic Regression, Random Forest, Gradient Boosting, and TensorFlow Wide & Deep) to accurately predict churn probability.
- **Business Intelligence:** Built an interactive web dashboard providing executive summaries, cohort analysis, risk segmentation, and actionable recommendations.

---

## 🏗️ Architecture & Tech Stack

1. **ETL & Processing (`src/pulse/pipeline.py`)**
   - **Pandas** for vectorized data manipulation, cleaning, and feature engineering.
   - Handles missing values, outliers, and normalizes engagement metrics.
   
2. **Data Warehouse (`sql/schema.sql`)**
   - **SQLite3** acts as a local, serverless data warehouse.
   - Stores raw facts (`fact_subscribers`) and aggregated KPI views (`sql/views/kpi_views.sql`).
   
3. **Machine Learning (`src/pulse/models/`)**
   - **Scikit-learn** for baseline and ensemble models.
   - **TensorFlow** for advanced deep learning approaches.
   - Automated hyperparameter tuning and model evaluation (ROC-AUC, Precision/Recall).

4. **Interactive Dashboard (`src/pulse/dashboard.py`)**
   - **Streamlit** powered frontend mimicking the premium *Spark Admin* UI theme (Dark Forest & Lime accents).
   - **Plotly Express** for highly interactive, colorblind-safe visualizations.

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.10 or higher installed.

### 2. Installation
Clone this repository and set up a virtual environment:

```bash
git clone https://github.com/YOUR_USERNAME/pulse-retention-intelligence.git
cd pulse-retention-intelligence

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install -e .
```

### 3. Run the Data Pipeline
Execute the ETL and Model Training pipeline. This will generate synthetic data, build the SQLite warehouse, and train the ML models:
```bash
python -m pulse.pipeline
```

### 4. Launch the Dashboard
Fire up the interactive Streamlit dashboard to explore the insights:
```bash
python -m streamlit run src/pulse/dashboard.py
```
*(The dashboard will automatically open in your default web browser at `http://localhost:8501`)*

---

## 📊 Dashboard Features

- **Executive Overview:** High-level metrics tracking Total ARR, Active ARR, Churn Rate, and Average Engagement.
- **Trend & Cohort Analysis:** Monthly ARR movement and cohort retention heatmaps identifying historical drop-off points.
- **Risk Segmentation:** Scatter plots and bar charts breaking down churn by Acquisition Channel, Plan Tier, and Geographic Region.
- **Risk Watchlist:** An actionable, filterable table of specific accounts flagged as High/Medium risk by the Machine Learning model.
- **Model Reliability:** Transparent breakdown of model evaluation metrics (ROC-AUC, F1-Score) and prediction distributions.

---

## 🧪 Testing

The codebase maintains strict quality standards with a comprehensive test suite. To run the tests:
```bash
pytest tests/
```

---

## 📝 License
This project was developed as a portfolio piece to demonstrate Data Analytics and Data Science proficiencies. Feel free to fork and use it for learning purposes!
