# ☀️ Smart Blinds IoT Monitoring System
## 👨‍🏫 Supervisor
Prof. Mehmet Ali Akyol  
Course: IoT and Digital twins   
Project: Smart Shading System


An interactive **Streamlit dashboard** for monitoring IoT sensor data related to
**light intensity** and **temperature**, designed to support intelligent control
of smart blinds for improved comfort and energy efficiency.

---

## 📌 Project Objectives
- Monitor environmental conditions using IoT sensor data
- Analyze the relationship between light intensity and temperature
- Display the current status of smart blinds (Open / Closed)
- Provide interactive visualizations for better decision-making

---

## 🛠️ Technologies Used
- **Python**
- **Streamlit** – dashboard and UI
- **Pandas** – data processing
- **Plotly** – interactive charts
- **Scikit-learn** – simple prediction model
- **OpenPyXL** – reading Excel files

---

## 📂 Project Structure
Smart-Blinds-Dashboard/
│
├── app.py
├── requirements.txt
├── README.md
└── data/
└── iot final.xlsx

---

## ▶️ How to Run the Project Locally

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt

2️⃣ Run the Streamlit app

streamlit run app.py

3️⃣ Open in browser

http://localhost:8501


## 🌐 Live Deployment (AWS EC2)

The Streamlit dashboard is deployed on an AWS EC2 Free Tier instance.

🔗 Live URL:
http://51.21.202.13:8501

Note: The application is hosted on an EC2 Ubuntu server and exposed via port 8501.

📊 Dataset Description

The dataset contains IoT sensor readings with the following attributes:

Timestamp – date and time of measurement

Temperature (°C) – ambient temperature

Light Intensity (Lux) – measured light level

Blind Status – current state of the blinds (Open / Closed)


✅ Results

The dashboard provides:

Interactive time-series analysis

Relationship visualization between temperature and light

Blind status filtering

Baseline temperature prediction from light intensity

Clear insights to support smart blind automation


🚀 Future Improvements

Real-time sensor data integration (MQTT / API)

Advanced ML models for blind automation

User authentication

Mobile-friendly UI


👤 Author

ِAbdulrahman Almaamari
Sam Almahdi
Mogdam Hayder

IoT & Data Visualization Project

