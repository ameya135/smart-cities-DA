# HVAC System Optimization with GRU

This repository contains the code and resources for optimizing HVAC (Heating, Ventilation, and Air Conditioning) system parameters in smart city environments. The focus is on leveraging machine learning techniques, particularly GRU (Gated Recurrent Unit), to forecast energy consumption and improve energy efficiency. This is part of our Design of Smart cities course, as a graded project.

## Project Overview

- **Objective**: Forecasting HVAC parameters such as heating power and optimize energy consumption in large public buildings.
- **Dataset**: Time-series data including temperature, humidity, heating power, outside temperature, and wind speed.
- **Model Used**: GRU12, chosen for its efficiency and ability to handle sequential data with temporal dependencies.

## Methodology

1. **Data Preprocessing**: Includes normalization and formatting for sequential input.
2. **Model Training**:
   - GRU architecture with 50 units.
   - Optimized using Adam optimizer and Mean Squared Error loss function.
3. **Evaluation**:
   - Metrics: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and accuracy.

## Results

| **Model** | **MSE** | **RMSE** | **Accuracy (%)** |
|-----------|---------|----------|------------------|
| GRU       | 0.025   | 0.158    | 98%              |
| LSTM      | 0.030   | 0.173    | 97%              |
| ARIMA     | 0.045   | 0.212    | 92%              |

## Project Structure

- `main.py`: Main script to train and evaluate the GRU model for temperature forecasting
- `rnn.py`: Contains the implementation of RNN classes (GRU, LSTM, VanillaRNN) and utility functions
- `optimization.py`: Particle swarm optimization to find optimal control parameters
- `data_example.csv`: Example dataset for training and testing
- `requirements.txt`: List of required Python packages

## How to Run

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Train the temperature prediction model:
   ```
   python main.py
   ```

3. Run optimization to find optimal control parameters:
   ```
   python optimization.py
   ```

## Contributors

- Kushagra Sahay, Utkarsh Arya, Ameya Avinash Patil, Rishabh Pandey
# smart-cities-DA
