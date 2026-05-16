# EDS_[TUPM-25-0164]_[Dela Cruz]
## REN-01: Wind Turbine Power Curve Deviation Analysis
**Pillar 4 — Renewable Energy Systems**

---

## Project Overview
This project implements a production-grade Python data analytics pipeline to analyze **Power Curve Deviation** in wind turbine SCADA data. The pipeline covers the full engineering workflow: data ingestion → cleaning → statistical analysis → visualization → reporting.

**Dataset:** [Wind Turbine SCADA Dataset — Kaggle (berkerisen)](https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset)

**Unique Filter Applied:** `Month == 1` (January only) — isolates winter low-temperature performance characteristics, where icing and cold-weather losses have the greatest impact on power curve deviation.

---

## Repository Structure
```
EDS_[TUPM-25-0164]_[Dela Cruz]/
│
├── main.py                   # Full Python analytics pipeline
├── requirements.txt          # Required Python libraries
├── README.md                 # This file
│
├── data/
│   ├── dataset_original.csv  # Raw SCADA data (auto-generated or Kaggle download)
│   └── dataset_cleaned.csv   # Cleaned, filtered dataset used for analysis
│
└── outputs/
    ├── static1_power_curve_scatter.png
    ├── static2_deviation_histogram.png
    ├── static3_wind_bin_boxplot.png
    ├── static4_correlation_heatmap.png
    ├── static5_hourly_profile.png
    └── analytics_report.md
```

---

## Pipeline Architecture (5 OOP Modules)

| Class | Role |
|---|---|
| `DataIngestion` | Loads CSV via local path or auto-downloads via kagglehub; falls back to synthetic data |
| `DataCleaner` | Deduplication, missing value handling, range correction, feature engineering, unique filter |
| `DataAnalyzer` | NumPy-based descriptive stats, distribution, correlation, comparative, and regression analysis |
| `Visualizer` | Generates 5 publication-ready static plots |
| `ReportGenerator` | Prints structured console report and writes `analytics_report.md` |

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/[YourUsername]/EDS_[StudentNumber]_[Surname].git
cd EDS_[StudentNumber]_[Surname]
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Set up Kaggle credentials for real dataset
- Go to [kaggle.com](https://www.kaggle.com) → Account → API → **Create New Token**
- Place `kaggle.json` in:
  - **Windows:** `C:\Users\<YourName>\.kaggle\kaggle.json`
  - **Mac/Linux:** `~/.kaggle/kaggle.json`

> If credentials are not set up, the pipeline automatically falls back to a realistic synthetic dataset and still runs completely.

### 4. Run the pipeline
```bash
python main.py
```

### 5. View outputs
All plots and the analytics report are saved to the `outputs/` folder automatically.

---

## Key Analytical Findings

| Metric | Value | Interpretation |
|---|---|---|
| Mean Power Deviation | −20.4 kW | Turbine under-performs theoretical curve in January |
| Skewness | −4.03 | Left-skewed; fault events dominate downward deviations |
| IQR Outliers | 1,483 | Significant anomalies during winter operation |
| Pearson r (WS vs Dev) | −0.41 | Moderate negative correlation — deviation worsens at high wind speeds |
| Polynomial R² | 0.88 | Model explains 88% of variance in actual power output |

---

## Libraries Used
| Library | Purpose |
|---|---|
| `numpy` | Descriptive statistics, numerical transformations |
| `pandas` | Data ingestion, cleaning, feature engineering |
| `matplotlib` | Static visualizations |
| `seaborn` | Correlation heatmap |
| `scipy` | Distribution tests, regression, non-parametric tests |
| `kagglehub` | Automatic dataset download |

---

## IEEE Paper Title
*Wind Turbine Power Curve Deviation Analysis Using SCADA-Based Modeling*
