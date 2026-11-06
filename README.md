# Last-Mile Delivery Route Optimization and Analytics

Week 1 task for the Yuva Intern data science program: **Strategic Planning and Data Exploration in Logistics**.

## About the project
This project plans how a courier company (a made-up company called QuickCart Logistics) could use data science to make last-mile deliveries faster, cheaper and more reliable.

The scenario: 3 hubs, about 60 vans and around 8,000 parcels a day. The company wants to cut costs, reduce late deliveries and predict demand better.

## What is in this repository
| File | Description |
|------|-------------|
| `Week1_Logistics_Report_Humanized.docx` | The full strategic planning report |
| `route_analysis.py` | Python code illustrating the approach |
| `requirements.txt` | Python libraries needed |

## KPIs tracked
- On-Time Delivery Rate
- Cost per Delivery
- Average Distance per Parcel
- First-Attempt Success Rate
- Vehicle Utilization

## Approach
1. **Data collection and cleaning**: load orders, remove duplicates and bad coordinates, create delay targets.
2. **Exploratory analysis**: late deliveries by hour, correlations, KPI calculation.
3. **Regression**: predict delivery time with Gradient Boosting.
4. **Clustering**: split the city into delivery zones with K-Means.
5. **Optimization**: plan routes as a Vehicle Routing Problem with Google OR-Tools.
6. **Forecasting**: predict daily demand with Prophet.

## How to run
```bash
pip install -r requirements.txt
python route_analysis.py
```
The script creates **synthetic (made-up) data** so it runs without any dataset. For a real project, replace the data step with files such as the Olist Brazilian E-Commerce dataset or the Amazon Last Mile Routing dataset.

## Data sources suggested in the report
- World Bank Logistics Performance Index
- Olist Brazilian E-Commerce dataset (Kaggle)
- Amazon Last Mile Routing Research Challenge dataset
- OpenStreetMap

## Note
This is a planning and illustration project. KPI baselines and targets in the report are assumptions, not measured results.
