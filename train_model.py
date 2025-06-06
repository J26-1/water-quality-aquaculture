import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
import os

# Load dataset
df = pd.read_csv("brisbane_water_quality.csv")

# Clean column names
df.columns = df.columns.str.replace(r"[\[\]]", "", regex=True)  # remove [ and ]

features = ['Timestamp', 'Record number', 'Average Water Speed', 'Average Water Direction',
            'Chlorophyll', 'Chlorophyll quality', 'Temperature', 'Temperature quality',
            'Dissolved Oxygen', 'Dissolved Oxygen quality', 'Dissolved Oxygen %Saturation',
            'Dissolved Oxygen %Saturation quality', 'pH quality', 'Salinity', 'Salinity quality',
            'Specific Conductance', 'Specific Conductance quality', 'Turbidity', 'Turbidity quality']

target = 'pH'


X = df[features].copy()
y = df[target]

# Convert Timestamp to datetime and extract features
X['Timestamp'] = pd.to_datetime(X['Timestamp'])
X['Timestamp_unix'] = X['Timestamp'].astype('int64') // 10**9
X['year'] = X['Timestamp'].dt.year
X['month'] = X['Timestamp'].dt.month
X['day'] = X['Timestamp'].dt.day
X['hour'] = X['Timestamp'].dt.hour
X['minute'] = X['Timestamp'].dt.minute
X['weekday'] = X['Timestamp'].dt.weekday
X = X.drop(columns=['Timestamp'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model folder if not exist
if not os.path.exists('model'):
    os.makedirs('model')

# Train model
model = lgb.LGBMRegressor()
model.fit(X_train, y_train)

# Save model (both pickle and LightGBM native)
joblib.dump(model, "model/lightgbm_model.pkl")
model.booster_.save_model("model/lightgbm_model.txt")

print("Model saved to model/lightgbm_model.pkl and model/lightgbm_model.txt")


