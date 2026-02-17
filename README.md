# 🚲 Urban Mobility: Bike Rental Demand Prediction
### Microsoft Elevate AICTE Internship Capstone Project

## 📝 Overview
This project addresses the challenge of predicting hourly bike rental demand using **Azure Machine Learning**. By forecasting demand accurately, city mobility systems can ensure a stable supply of bikes and reduce user wait times.

## 🚀 System Features
- **Data Analytics:** Analyzing historical patterns based on weather and time.
- **Machine Learning:** Utilizing Random Forest Regressor for time-series forecasting.
- **Azure Integration:** Trained and deployed using Azure ML SDK v2.

## 📂 Repository Structure
- `src/`: Contains the training script (`train.py`).
- `azure-ml/`: Azure job configuration (`job.yml`).
- `requirements.txt`: List of dependencies.

## 🛠️ Instructions
1. Upload the `bike-no.csv` dataset to your Azure ML Workspace.
2. Configure a Compute Cluster in Azure ML Studio.
3. Run the job using the Azure CLI: `az ml job create -f azure-ml/job.yml`.
4. Monitor metrics in the **Experiments** tab.
