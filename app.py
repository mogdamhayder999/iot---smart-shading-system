import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# -----------------------------
# Load and preprocess data
# -----------------------------
st.set_page_config(page_title="Smart Blinds Dashboard", layout="wide")
st.title("☀️ Smart Blinds IoT Monitoring System")
st.write("Monitoring Light Intensity and Temperature with Blind Status")

df = pd.read_excel("data/iot final.xlsx")

# Rename columns
df = df.rename(columns={
    'Temp (°C)': 'temp',
    'Light (Lux)': 'light_intensity_lux',
    'Blind Status': 'blind_status',
    'Sun Elevation': 'sun_elevation',
    'Timestamp': 'timestamp'
})

df['timestamp'] = pd.to_datetime(df['timestamp'])

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Filters")
selected_status = st.sidebar.multiselect(
    "Select Blind Status",
    options=df['blind_status'].unique(),
    default=df['blind_status'].unique()
)
filtered_df = df[df['blind_status'].isin(selected_status)]

# -----------------------------
# Summary statistics
# -----------------------------
st.subheader("Summary Statistics")
st.write(filtered_df[['temp','light_intensity_lux']].describe())

# -----------------------------
# Time series plot
# -----------------------------
st.subheader("Temperature and Light Intensity Over Time")
fig_time = px.line(filtered_df, x='timestamp', y=['temp','light_intensity_lux'],
                   labels={'value':'Value','timestamp':'Time'},
                   title="Temperature and Light Intensity Time Series")
st.plotly_chart(fig_time)

# -----------------------------
# Scatter plot
# -----------------------------
st.subheader("Relationship between Light Intensity and Temperature")
fig_scatter = px.scatter(filtered_df, x='light_intensity_lux', y='temp',
                         color='blind_status',
                         title="Light vs Temperature by Blind Status",
                         labels={'light_intensity_lux':'Light Intensity (Lux)',
                                 'temp':'Temperature (°C)'})
st.plotly_chart(fig_scatter)

# -----------------------------
# Correlation matrix
# -----------------------------
st.subheader("Correlation Matrix")
corr_matrix = filtered_df[['temp','light_intensity_lux']].corr()
st.write(corr_matrix)

# -----------------------------
# Baseline Prediction Model
# -----------------------------
st.subheader("Predict Temperature from Light Intensity")
X = filtered_df[['light_intensity_lux']]
y = filtered_df['temp']

if len(filtered_df) > 1:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
    st.write("Sample Predictions:")
    st.write(pd.DataFrame({"Actual": y_test, "Predicted": y_pred}).head())
else:
    st.write("Not enough data for prediction.")

# -----------------------------
# Missing values
# -----------------------------
st.subheader("Missing Values per Column")
st.write(filtered_df.isnull().sum())
