# 🌊 Water Quality Monitoring System for Aquaculture in Malaysia

## 📌 Overview

This is a real-time, web-based water quality monitoring system tailored specifically for **Malaysian aquaculture farms**, particularly **small and medium enterprises (SMEs)**. By integrating affordable **IoT sensors** with a **Flask-based backend**, **LightGBM-powered predictions**, and a **JavaScript frontend**, this project helps farmers monitor vital water parameters such as:

- Temperature 🌡️  
- pH 🧪  
- Dissolved Oxygen 💧  
- Ammonia Levels 🐟  

Farmers can access live data, receive alerts, and visualize historical trends – all from a simple, mobile-friendly web dashboard.

---

## 🎯 Project Goals

- Provide an **affordable**, **localized**, and **scalable** system for small-scale aquaculture operators in Malaysia.
- Enable **real-time** monitoring with **custom alerts** for different species/farm conditions.
- Include **AI-powered predictive analytics** for smarter decision-making.
- Ensure **ease of use** through a clean, intuitive web interface.

---

## 🚨 Problem Statement

Malaysian SME aquaculture farmers often rely on outdated and manual water testing methods that:
- Are **labor-intensive** and error-prone  
- Lack **real-time** capability  
- Cannot provide **data-driven** insights or forecasting  
- Struggle with **environmental volatility** (e.g., heavy rains, heatwaves)

This project fills the gap by offering a **cost-effective**, **localized**, and **user-centered** solution that empowers farmers to improve water quality, reduce losses, and boost productivity.

---

## 📦 Tech Stack

| Layer       | Technology         |
|-------------|--------------------|
| Frontend    | JavaScript         |
| Backend     | Flask (Python)     |
| Machine Learning | LightGBM      |
| Database    | JavaScript-based storage (e.g., IndexedDB/localStorage during prototype) |
| Deployment  | Render             |

---

## 🧠 AI & Analytics

- Integrated **LightGBM** model trained on sample aquaculture datasets
- Provides **predictive insights** (e.g., trend forecasting for dissolved oxygen or ammonia levels)
- Customizable thresholds based on specific **species or pond types**

---

## 🖥️ Features

- ✅ Real-time sensor data collection via IoT modules  
- ✅ Live data dashboard (temperature, pH, DO, ammonia)  
- ✅ Custom threshold alerts & push notifications  
- ✅ AI-driven trend prediction and water quality suggestions  
- ✅ Responsive and mobile-friendly web UI  
- ✅ Historical data visualization & export  

---

## 👥 Target Users

- 🎣 **Small & Medium-Sized Aquaculture Farmers**  
- 🧑‍🔬 **Aquaculture Consultants / Researchers**  
- 🏢 **Government Agencies / Policy Makers**  
- 🧪 **Environmental Scientists & Institutions**  

---

## 🔬 System Architecture

```plaintext
IoT Sensors → Flask Backend API → LightGBM Model → JavaScript Frontend
                                ↓
                     Render Deployment (CI/CD)
