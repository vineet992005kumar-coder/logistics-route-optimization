"""
Last-Mile Delivery Route Optimization and Analytics
Week 1 Task - Yuva Intern

Illustrates: data cleaning, EDA, regression, clustering,
route optimization (OR-Tools) and demand forecasting (Prophet).
Runs on synthetic data so no dataset is needed.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)


# ---------------------------------------------------------------
# 1. Create sample data (replace with real data in a real project)
# ---------------------------------------------------------------
def make_sample_data(n=3000):
    start = pd.Timestamp("2025-01-01")
    order_time = start + pd.to_timedelta(rng.integers(0, 90 * 24 * 60, n), unit="m")
    distance = rng.gamma(2.0, 3.0, n)            # km
    n_stops = rng.integers(5, 40, n)
    weight = rng.gamma(2.0, 2.0, n)              # kg
    rain = rng.integers(0, 2, n)
    minutes = 20 + 4 * distance + 0.8 * n_stops + 1.5 * weight + 10 * rain + rng.normal(0, 8, n)
    df = pd.DataFrame({
        "order_id": np.arange(n),
        "order_time": order_time,
        "promised_time": order_time + pd.to_timedelta(90, unit="m"),
        "delivered_time": order_time + pd.to_timedelta(minutes, unit="m"),
        "dest_lat": 12.9 + rng.normal(0, 0.05, n),
        "dest_lon": 77.6 + rng.normal(0, 0.05, n),
        "distance_km": distance,
        "n_stops": n_stops,
        "weight_kg": weight,
        "is_rain": rain,
    })
    return df


# ---------------------------------------------------------------
# 2. Cleaning
# ---------------------------------------------------------------
def clean(orders):
    orders = orders.drop_duplicates(subset="order_id")
    orders = orders.dropna(subset=["delivered_time", "dest_lat", "dest_lon"])
    orders = orders[orders["dest_lat"].between(-90, 90) & orders["dest_lon"].between(-180, 180)]

    orders["delivery_minutes"] = (
        (orders["delivered_time"] - orders["order_time"]).dt.total_seconds() / 60
    )
    orders["is_late"] = (orders["delivered_time"] > orders["promised_time"]).astype(int)

    cap = orders["delivery_minutes"].quantile(0.99)       # treat outliers
    orders["delivery_minutes"] = orders["delivery_minutes"].clip(upper=cap)
    orders["hour"] = orders["order_time"].dt.hour
    orders["day_of_week"] = orders["order_time"].dt.dayofweek
    return orders


# ---------------------------------------------------------------
# 3. Exploratory analysis and KPI
# ---------------------------------------------------------------
def explore(orders):
    otd = 100 * (1 - orders["is_late"].mean())
    print(f"On-Time Delivery Rate: {otd:.1f}%")
    print(f"Average delivery time: {orders['delivery_minutes'].mean():.1f} min")

    late_by_hour = orders.groupby("hour")["is_late"].mean() * 100
    late_by_hour.plot(kind="bar", title="Late Delivery % by Hour of Day")
    plt.ylabel("% late")
    plt.tight_layout()
    plt.savefig("late_by_hour.png")
    plt.close()


# ---------------------------------------------------------------
# 4. Regression: predict delivery time
# ---------------------------------------------------------------
def train_delivery_time_model(orders):
    features = ["distance_km", "n_stops", "weight_kg", "hour", "day_of_week", "is_rain"]
    X, y = orders[features], orders["delivery_minutes"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4)
    model.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, model.predict(X_test))
    print(f"Delivery time model - average error: {mae:.1f} minutes")
    return model


# ---------------------------------------------------------------
# 5. Clustering: delivery zones
# ---------------------------------------------------------------
def make_zones(orders, k=12):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    orders["zone"] = kmeans.fit_predict(orders[["dest_lat", "dest_lon"]])
    summary = orders.groupby("zone").agg(
        parcels=("order_id", "count"),
        avg_delivery_min=("delivery_minutes", "mean"),
        late_rate=("is_late", "mean"),
    )
    print("\nZones with highest late rate:")
    print(summary.sort_values("late_rate", ascending=False).head(5))
    return orders


# ---------------------------------------------------------------
# 6. Optimization: Vehicle Routing Problem (OR-Tools)
# ---------------------------------------------------------------
def solve_vrp(dist_matrix, num_vehicles, depot=0, capacity=100, demands=None):
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    manager = pywrapcp.RoutingIndexManager(len(dist_matrix), num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    def distance_cb(i, j):
        return dist_matrix[manager.IndexToNode(i)][manager.IndexToNode(j)]

    transit = routing.RegisterTransitCallback(distance_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    def demand_cb(i):
        return demands[manager.IndexToNode(i)]

    demand = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(demand, 0, [capacity] * num_vehicles, True, "Capacity")

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = 5

    solution = routing.SolveWithParameters(params)
    if not solution:
        print("No route solution found.")
        return None

    for v in range(num_vehicles):
        index, route, dist = routing.Start(v), [], 0
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            nxt = solution.Value(routing.NextVar(index))
            dist += routing.GetArcCostForVehicle(index, nxt, v)
            index = nxt
        route.append(0)
        print(f"Vehicle {v}: route {route}, distance {dist}")
    return solution


def demo_routing():
    # Small demo: 1 depot + 8 stops, 2 vans
    pts = rng.uniform(0, 20, (9, 2))
    dist = (np.sqrt(((pts[:, None] - pts[None, :]) ** 2).sum(-1)) * 100).astype(int).tolist()
    demands = [0] + [10] * 8
    print("\nRoute optimization demo:")
    solve_vrp(dist, num_vehicles=2, capacity=50, demands=demands)


# ---------------------------------------------------------------
# 7. Forecasting: daily demand (Prophet)
# ---------------------------------------------------------------
def forecast_demand(orders):
    from prophet import Prophet

    daily = orders.groupby(orders["order_time"].dt.date).size().reset_index()
    daily.columns = ["ds", "y"]
    daily["ds"] = pd.to_datetime(daily["ds"])

    m = Prophet(weekly_seasonality=True, yearly_seasonality=False)
    m.fit(daily)
    future = m.make_future_dataframe(periods=14)
    forecast = m.predict(future)
    m.plot(forecast)
    plt.savefig("demand_forecast.png")
    plt.close()
    print("\nSaved 14-day demand forecast to demand_forecast.png")


# ---------------------------------------------------------------
if __name__ == "__main__":
    orders = clean(make_sample_data())
    explore(orders)
    train_delivery_time_model(orders)
    orders = make_zones(orders)

    try:
        demo_routing()
    except ImportError:
        print("Install ortools to run the routing demo: pip install ortools")

    try:
        forecast_demand(orders)
    except ImportError:
        print("Install prophet to run the forecast: pip install prophet")
