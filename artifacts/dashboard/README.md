# Tradier Stock Market Dashboard - Capstone Project

## Overview
This dashboard presents the empirical results from a capstone project on short-horizon equity prediction using machine learning. The analysis covers Tradier 2024 data (Jan-Dec, full year) using time-aware, leakage-safe evaluation.

## Features

### 📊 Market Data Tab
- Interactive stock symbol selection and date range filtering
- Real-time price charts (Candlestick and Line charts)
- Trading volume visualization
- Price prediction with confidence bands
- Statistical summaries and raw data tables
- Multi-symbol comparison views

### 🔬 Empirical Results Tab
- **Prediction Accuracy Summary**: Key metrics (51.9% accuracy, 0.526 ROC-AUC)
- **Research Questions & Findings**: Answers to three key research questions
- **Classification Results**: Next-bar direction prediction results
- **Regression Results**: Next-bar return prediction results
- **Feature Engineering**: Comprehensive documentation of feature construction
- **Evaluation Design**: Methodology and reproducibility details

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Ensure `tradier_math.csv` is in the same directory as `dashboard.py`

## Running the Dashboard

```bash
streamlit run dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Project Structure

```
Dashboard-2024-TradierData/
├── dashboard.py              # Main dashboard application
├── tradier_math.csv          # Stock market data
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Key Results

- **Classification**: Random Forest achieved ROC-AUC = 0.526 and Accuracy = 0.519
- **Lift Over Random**: Modest but consistent ~2.6% improvement
- **Methodology**: Expanding walk-forward cross-validation, leakage-safe features
- **Features**: 16 top features identified via correlation analysis

## Technologies Used

- Streamlit for dashboard framework
- Plotly for interactive visualizations
- Pandas for data manipulation
- NumPy for numerical computations

## Capstone Project Details

- **Data Source**: Tradier 2024 (Jan-Dec, full year)
- **Evaluation**: Time-aware, leakage-safe protocol
- **Models**: Logistic Regression, Decision Tree, Random Forest (classification)
- **Models**: Ridge, ElasticNet, Random Forest Regressor (regression)

## Contact

For questions about this capstone project, please refer to the ShortHorizonEquityPrediction document.

