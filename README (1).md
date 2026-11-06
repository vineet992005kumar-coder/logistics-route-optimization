# Week 2: Data Collection, Cleaning and Preprocessing for Logistics Analysis

Yuva Intern data science task. This folder simulates a messy logistics dataset and cleans it with pandas.

## Files
| File | Description |
|------|-------------|
| `Week2_Data_Cleaning_Preprocessing_Report.docx` | Full report |
| `data_preprocessing.py` | Python pipeline: simulate data, clean, preprocess |

## What the script does
1. Simulates 5,000 orders with realistic problems (duplicates, missing values, outliers, impossible values, inconsistent labels)
2. Prints a data quality report before and after cleaning
3. Removes duplicates, fixes labels and impossible values
4. Fills missing values (mode for categories, median per hub for numbers)
5. Caps extreme outliers with the 3 x IQR rule
6. Creates features, applies Min-Max and z-score scaling, and one-hot encodes categories
7. Saves `logistics_clean.csv`

## Run
```bash
pip install pandas numpy scikit-learn
python data_preprocessing.py
```

Note: data is simulated for learning purposes, with structure inspired by the Olist Brazilian E-Commerce dataset.
