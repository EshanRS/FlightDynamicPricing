from flask import Flask, render_template, request
import joblib
import pandas as pd

# Load saved model pipeline
model = joblib.load("random_forest_price_model.pkl")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction_text = None

    if request.method == "POST":
        source = request.form["source"]
        destination = request.form["destination"]
        date = request.form["date"]      # yyyy-mm-dd format
        time = request.form["time"]      # HH:MM (24hr format)

        # Extract day, month, year, hour, minute
        date_parts = date.split("-")
        year = int(date_parts[0])
        month = int(date_parts[1])
        day = int(date_parts[2])

        time_parts = time.split(":")
        hour = int(time_parts[0])
        minute = int(time_parts[1])

        # Build input dictionary — adjust column names to match your dataset
        input_data = {
            "Source": source,
            "Destination": destination,
            "Day": day,
            "Month": month,
            "Year": year,
            "Hour": hour,
            "Minute": minute
        }

        input_df = pd.DataFrame([input_data])
        # USER INPUTS
        source = request.form.get("source")
        destination = request.form.get("destination")
        date = request.form.get("date")          # yyyy-mm-dd
        time = request.form.get("time")          # HH:MM

        # Convert date/time
        input_datetime = pd.to_datetime(date + " " + time)

        # Generate a random time generator for arrival time
        import random
        random_minute = random.randint(0, 59)
        random_hour = random.randint(0, 23)
        arrival_time = f"{random_hour:02}:{random_minute:02}"

        # to chose randomly from duration options
        duration_options = ["2h 55m", "2h 50m","5h 35m"]
        duration = random.choice(duration_options)

        # To chose randomly from non-stop or 1 stop
        stops_options = ["non stop", "1 stop"]
        total_stops = random.choice(stops_options)

        # To chose from flight number randomly
        flight_numbers = ["6E 6813", "6E 6283", "6E 6021", "6E 6401"]
        flight_no = random.choice(flight_numbers)


        # STATIC VALUES FOR ALL OTHER FEATURES
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
             "Time_Stamp_Date": input_datetime.date(),
             "Time_Stamp_Time": input_datetime.time()
}

        # Create proper input DataFrame
        input_df = pd.DataFrame([static_values])

        # Predict
        pred = model.predict(input_df)[0]
        prediction_text = f"The predicted price for the given parameters time is ₹{pred:.2f}"

    return render_template("index.html", prediction=prediction_text)


if __name__ == "__main__":
    app.run(debug=True)
