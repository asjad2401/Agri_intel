# Agri-Intel: District-Wise Yield Optimization System
**CS-245 Machine Learning Project - Fall 2025**

##  Project Overview
Agri-Intel is a machine learning-based decision support system designed to optimize wheat production in Punjab. By integrating **Government Agricultural Statistics**, **MODIS Satellite Data (NDVI)**, and **NASA POWER Climate Data**, this system predicts crop yields and recommends optimal fertilizer usage for sustainable agriculture (Theme T4).

##  Project Structure
* `app.py`: The main Streamlit application (Proof of Concept Dashboard).
* `final_model.pkl`: The trained Gradient Boosting Machine Learning model.
* `final_dataset_ml_ready.csv`: The merged master dataset used for predictions.
* `files/punjab_districts_cleaned.geojson`: Spatial data for the interactive map.
* `Data_pipeline.ipynb`: Notebook containing data collection and processing scripts.
* `requirements.txt`: List of Python dependencies.

##  Installation
1. Ensure you have Python installed (version 3.8 or higher).
2. Install the required libraries using pip:
   ```bash
   pip install -r requirements.txt

## How to Run the Dashboard
Open your terminal in the project directory.

Run the following command:

`Bash`

`streamlit run app.py`
The dashboard will automatically open in your browser at http://localhost:8501.
or visit 
`https://asjad2401-agri-intel-streamlitapp-apuzxg.streamlit.app/`

## Usage Guide
Select District: Choose a district from the sidebar to load its historical soil and climate defaults.

Analyze Yield: Adjust the sliders (Fertilizer, NDVI, Rainfall) and click "Analyze Yield" to see the prediction.

Optimization Simulation: Scroll down to the "What-If" section to see the AI's recommendation for maximizing yield based on your inputs.

Developed by: Muhammad Asjad
