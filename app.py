
import os
import random
import matplotlib
matplotlib.use("Agg")   # use non-GUI backend for Flask
import matplotlib.pyplot as plt
from joblib import load
from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

# Ensure static directory exists
if not os.path.exists("static"):
    os.makedirs("static")

# Load saved model pipeline once
MODEL_PATH = "random_forest_price_model.pkl"
model = joblib.load("random_forest_price_model.pkl")

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction_text = None
    graph_path = None

    if request.method == "POST":
        # -------------------------
        # Read form inputs
        # -------------------------
        source = request.form.get("source", "Bengaluru")
        destination = request.form.get("destination", "New Delhi")
        date = request.form.get("date")      # expected yyyy-mm-dd
        time = request.form.get("time")      # expected HH:MM (24-hr)

        # Validate required fields
        if not date or not time:
            prediction_text = "Please provide both date and time."
            return render_template("index.html", prediction=prediction_text, graph=graph_path)

        # Parse datetime
        try:
            input_datetime = pd.to_datetime(f"{date} {time}")
        except Exception as e:
            prediction_text = f"Invalid date/time format: {e}"
            return render_template("index.html", prediction=prediction_text, graph=graph_path)

        # -------------------------
        # Build static/randomized features (keeps UI simple)
        # -------------------------
        # small randomness for arrival/duration/stops/flight_no to simulate variety
        random_minute = random.randint(0, 59)
        random_hour = random.randint(0, 23)
        arrival_time = f"{random_hour:02}:{random_minute:02}"

        duration_options = ["2h 55m", "2h 50m", "5h 35m"]
        duration = random.choice(duration_options)

        stops_options = ["non stop", "1 stop"]
        total_stops = random.choice(stops_options)

        flight_numbers = ["6E 6813", "6E 6283", "6E 6021", "6E 6401"]
        flight_no = random.choice(flight_numbers)

        static_values = {
            "Airline": "IndiGo",
            "Source": source,
            "Destination": destination,
            "Route": "BLR → DEL",
            "Dep_Time": time,
            "Arrival_Time": arrival_time,
            "Duration": duration,
            "Total_Stops": total_stops,
            "Flight_no": flight_no,
            # These will be overwritten per timestamp when generating history
            "Time_Stamp_Date": input_datetime.date(),
            "Time_Stamp_Time": input_datetime.time()
        }

        # -------------------------
        # Single-point prediction for the user-specified datetime
        # -------------------------
        input_df = pd.DataFrame([static_values])
        try:
            # Model expects whatever columns it was trained on; if needed, adjust here
            pred = model.predict(input_df)[0]
            prediction_text = f"The estimated price for {date} at {time} is ₹{pred:.2f}"
        except Exception as e:
            prediction_text = f"Prediction error: {e}"
            return render_template("index.html", prediction=prediction_text, graph=graph_path)

        # -------------------------
        # Generate 48-hour history at 3-hour intervals (17 points: t-48h ... t)
        # -------------------------
        # Use periods=17 and freq="3h" to guarantee consistent spacing
        time_range = pd.date_range(end=input_datetime, periods=17, freq="3h")

        history_prices = []
        true_final_pred = None

        # Pre-load model (already loaded globally as `model`) and avoid reloading inside loop
        for ts in time_range:
            temp_values = static_values.copy()
            temp_values["Time_Stamp_Date"] = ts.date()
            temp_values["Time_Stamp_Time"] = ts.time()
            temp_df = pd.DataFrame([temp_values])

            # Model raw prediction for this timestamp
            try:
                raw_pred = model.predict(temp_df)[0]
            except Exception as e:
                # If model raises due to unexpected columns, break and report
                prediction_text = f"Prediction error during history generation: {e}"
                return render_template("index.html", prediction=prediction_text, graph=graph_path)

            # Save the true model-prediction at the final timestamp (exact departure)
            if ts == input_datetime:
                true_final_pred = raw_pred

            # -------------------------
            # Apply time-based adjustments:
            # - Steeper increase very near departure
            # - Small off-peak discount (00:00-06:00)
            # - Very light randomness (±1-2%)
            # But ensure the final point will be replaced by the true model prediction later.
            # -------------------------
            hours_before = (input_datetime - ts).total_seconds() / 3600.0
            # clamp hours_before to [0,48] for safety
            hours_before = max(0.0, min(48.0, hours_before))

            # Steeper curve near departure: cubic-like ramp, scaled up to +20% near 0 hours
            steep_factor = 1.0 + (1.0 - (hours_before / 48.0))**2 * 0.2

            # Off-peak small discount
            offpeak_factor = 0.96 if ts.hour in range(0, 7) else 1.0

            # Very light randomness
            noise = np.random.uniform(0.99, 1.01)  # ±1%

            adjusted_price = raw_pred * steep_factor * offpeak_factor * noise
            history_prices.append(adjusted_price)

        # Ensure the final plotted point equals the raw model prediction at departure
        if true_final_pred is not None:
            history_prices[-1] = true_final_pred

        # -------------------------
        # Plot and save the graph
        # -------------------------
        plt.figure(figsize=(12, 6))
        plt.plot(time_range, history_prices, marker="o", linewidth=2)
        plt.title("Estimated Price Trend — Last 48 Hours (3-hour intervals)")
        plt.xlabel("Time")
        plt.ylabel("Estimated Price (₹)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.xticks(rotation=45)

        graph_filename = "price_48hr.png"
        graph_path = os.path.join("static", graph_filename)
        plt.tight_layout()
        plt.savefig(graph_path, bbox_inches="tight")
        plt.close()

    return render_template("index.html", prediction=prediction_text, graph=graph_path)


if __name__ == "__main__":
    app.run(debug=True)
