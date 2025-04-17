from rnn import MyGRU, MyLSTM, VanillaRNN, RNN
import pandas as pd
import os

# Current directory
current_dir = os.getcwd()

# Download the downsampled data frame from csv-file.
raw_data = pd.read_csv(os.path.join(current_dir, 'data_example.csv'))

# Initialize the model with the required parameters.
hvac_model = MyGRU(quant=['Inside_temperature'], seq=12, fut=0, parameters=['Outside_humidity',
              'Solar_irradiance',
              'CO2_concentration',
              'hours_sin',
              'hours_cos',
              'weekday_sin',
              'weekday_cos',
              'Domestic_water_network_1_primary_valve',
              'Domestic_water_network_2_primary_valve',
              'District_heat_temperature',
              'Outside_temperature_average',
              'Ventilation_network_1_temperature',
              'Ventilation_network_2_temperature',
              'Radiator_network_1_temperature',
              'Radiator_network_2_temperature'])

# Scale, split, and sequence the downsampled data frame.
x_train, y_train, x_test, y_test = hvac_model.preprocess(raw_data)

# Train the model using custom fit method. 
# Does hyperparameter optimization automatically in pre-defined search space. Comment row below, if you have already trained the model.
hvac_model.fit(X=x_train, y=y_train, epochs=20, max_trials=1)
hvac_model.save()

# Example of loading a previously trained model (uncomment to use)
# latest_model_dir = f'GRU_Inside_temperature_{hvac_model.date}'
# hvac_model.load(os.path.join(current_dir, latest_model_dir))

# Calculating prediction intervals
#rounds = 12     # Number of data instances to calculate prediction intervals to.

#for i in range(rounds):
    
    # Calculating prediction percentiles and saving them to a csv file.
    #hvac_model.prediction_interval(x_train, y_train, x_test[i])     # NB! The process is computationally intensive.

# Making test predictions with the RNN model.
#preds = hvac_model.model.predict(x_test)

# Loading prediction intervals from disk if available
#latest_model_dir = f'GRU_Inside_temperature_{hvac_model.date}'
#intervals_path = os.path.join(current_dir, latest_model_dir, 'pred_ints.csv')
#if os.path.exists(intervals_path):
#    low, up = hvac_model.load_intervals(intervals_path)

# Inverse target variables both for measured values and computed predictions.
#preds, y_test_inv = hvac_model.inv_target(x_test, preds, y_test)
#if 'low' in locals() and 'up' in locals():
#    low, up = hvac_model.inv_target(x_test, low, up)

# Plot model prediction alongside measured values.
# Add lower and upper intervals as arguments to plot them.
#hvac_model.plot_preds(preds, y_test_inv)