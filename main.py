from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, Response, session
from datetime import datetime
from send_alert import send_whatsapp_alert  # Ensure this is implemented
from flask import send_file
import pandas as pd
import lightgbm as lgb
import time
import random
import os
import io
import numpy as np


app = Flask(__name__)
app.secret_key = os.urandom(24)
users = {}

# =======================
# Load and preprocess data
# =======================
data = pd.read_csv("brisbane_water_quality.csv")


if 'Timestamp' in data.columns:
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data['Timestamp_unix'] = data['Timestamp'].astype('int64') // 10**9
    data['year'] = data['Timestamp'].dt.year
    data['month'] = data['Timestamp'].dt.month
    data['day'] = data['Timestamp'].dt.day
    data['hour'] = data['Timestamp'].dt.hour
    data['minute'] = data['Timestamp'].dt.minute
    data['weekday'] = data['Timestamp'].dt.weekday
    data = data.drop(columns=['Timestamp'])


# =======================
# ML Model loading
# =======================
target_col = 'pH'
if target_col in data.columns:
    target = data[target_col]
    features = data.drop(columns=[target_col])
else:
    features = data
    target = None


model_path = 'model/lightgbm_model.txt'
if os.path.exists(model_path):
    model = lgb.Booster(model_file=model_path)
elif target is not None:
    model = lgb.LGBMRegressor()
    model.fit(features, target)
    model.booster_.save_model(model_path)
else:
    raise Exception("Model not found and target column missing — cannot train.")


predictions = model.predict(features)


# =======================
# Simulate Streaming Prediction
# =======================
def stream_predictions():
    for value in predictions:
        yield f"data: {value:.2f}\n\n"
        time.sleep(0.1)


# =======================
# Thresholds
# =======================
THRESHOLDS = {
    "Temperature": {"min": 24, "max": 30},
    "pH": {"min": 7, "max": 9},
    "DO": {"min": 4.0},
    "Turbidity": {"max": 10},
    "Average Water Speed": {"min": 0, "max": 20},
    "Chlorophyll": {"max": 40}
}


# =======================
# Simulate Real-Time Data
# =======================
def generate_live_data():
    alert_messages = []
    now = datetime.now()
    parameters = list(THRESHOLDS.keys())


    simulated_data = []


    for param in parameters:
        # Simulated values based on parameter
        if param == "Temperature":
            current = round(random.uniform(20, 35), 2)
        elif param == "pH":
            current = round(random.uniform(6.0, 8.5), 2)
        elif param == "DO":
            current = round(random.uniform(3.0, 8.0), 2)
        elif param == "Turbidity":
            current = round(random.uniform(2.0, 12.0), 2)
        elif param == "Average Water Speed":
            current = round(random.uniform(0, 25), 2)
        elif param == "Chlorophyll":
            current = round(random.uniform(10, 50), 2)
        else:
            current = round(random.uniform(0, 100), 2)


        predicted = round(current + random.uniform(-1, 1), 2)


        threshold = THRESHOLDS.get(param, {})
        min_val = threshold.get("min")
        max_val = threshold.get("max")


        current_status = "Normal"
        if (min_val and current < min_val) or (max_val and current > max_val):
            current_status = "Warning"


        predicted_status = "Normal"
        if (min_val and predicted < min_val) or (max_val and predicted > max_val):
            predicted_status = "Warning"


        simulated_data.append({
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "parameter": param,
            "current_value": current,
            "predicted_value_3h": predicted,
            "threshold": f"{min_val or ''}–{max_val or ''}".strip("–") if min_val or max_val else "N/A",
            "current_status": current_status,
            "predicted_status": predicted_status
        })


        if current_status == "Warning":
            alert_messages.append(f"🚨 CURRENT ALERT: {param} = {current} is outside threshold!")
        if predicted_status == "Warning":
            alert_messages.append(f"⏳ PREDICTION ALERT: {param} will be {predicted} in 3h")


    # Send alerts (e.g. to WhatsApp)
    for msg in alert_messages:
        send_whatsapp_alert(msg)


    return simulated_data


# =======================
# Flask Routes
# =======================
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if username in users:
            flash('Username already exists', 'error')
        elif password != confirm_password:
            flash('Passwords do not match', 'error')
        else:
            users[username] = password
            flash('Account created successfully. Please log in.', 'success')
            return redirect(url_for('login'))
    return render_template('signup.html')

@app.route("/dashboard_summary")
def dashboard_summary():
    return jsonify(generate_live_data())


@app.route('/ph-stream')
def ph_stream():
    return Response(stream_predictions(), mimetype='text/event-stream')

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        flash('You must be logged in to view the dashboard.', 'error')
        return redirect(url_for('login'))
    return render_template('dashboard.html', username=session['username'])

@app.route('/historical')
def historical():
    return render_template('historical.html')


@app.route('/download-historical-csv')
def download_historical_csv():
    try:
        start = request.args.get('start')
        end = request.args.get('end')


        df = pd.read_csv('brisbane_water_quality.csv')
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])


        if start and end:
            df = df[(df['Timestamp'] >= start) & (df['Timestamp'] <= end)]


        # Send CSV as attachment
        csv_io = io.StringIO()
        df.to_csv(csv_io, index=False)
        csv_io.seek(0)


        return Response(    
            csv_io.getvalue(),
            mimetype='text/csv',
            headers={"Content-Disposition": "attachment;filename=filtered_data.csv"}
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500




@app.route('/api/historical-data')
def get_historical_data():
    import json
    from flask import Response
    try:
        df = pd.read_csv('brisbane_water_quality.csv', parse_dates=['Timestamp'])


        start = request.args.get('start')
        end = request.args.get('end')


        print(f"Start: {start}, End: {end}")


        if start and end:
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)
            df = df[(df['Timestamp'] >= start_dt) & (df['Timestamp'] <= end_dt)]


        print(f"Filtered rows: {len(df)}")


        df = df.where(pd.notnull(df), None)
        data_dict = df.to_dict(orient='records')

        # Replace NaN with None (so it's valid JSON)
        df = df.replace({np.nan: None})

        return jsonify(df.to_dict(orient='records'))


        json_str = json.dumps(data_dict, default=str)
        return Response(json_str, mimetype='application/json')


    except Exception as e:
        print(f"Error loading historical data: {e}")
        return jsonify({'error': str(e)}), 500


# =======================
# Run Server
# =======================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

