from rnn import MyGRU, MyLSTM, VanillaRNN, RNN
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
import random
from datetime import datetime, date
import os

# Current directory
current_dir = os.getcwd()

# Load dataset first for both models to use
data_path = os.path.join(current_dir, 'data_example.csv')
data = pd.read_csv(data_path)

ener_model = MyGRU(quant=['Energy_consumption'], seq=12, fut=0, parameters=['Outside_humidity',
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
              
temp_model = MyGRU(quant=['Inside_temperature'], seq=12, fut=0, parameters=['Outside_humidity',
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

# DEFINE PATHS FOR LOADING ENERGY AND TEMPERATURE MODELS
# Use the latest trained models from the current directory
today = date.today()
ener_path = os.path.join(current_dir, f'GRU_Energy_consumption_{today}')
temp_path = os.path.join(current_dir, f'GRU_Inside_temperature_{today}')

# Function to train a model if it doesn't exist
def ensure_model_exists(model, model_path, model_type):
    if not os.path.exists(model_path) or not os.path.exists(os.path.join(model_path, 'model.h5')):
        print(f"{model_type} model not found. Training new model...")
        # Process data
        X_train, y_train, X_val, y_val = model.preprocess(raw_data=data)
        # Train with a small number of epochs and trials for quick results
        model.fit(X=X_train, y=y_train, epochs=20, max_trials=1)
        model.save()
        return True
    else:
        print(f"Loading existing {model_type} model...")
        model.load(model_path)
        return False

# Ensure both models exist
ener_trained = ensure_model_exists(ener_model, ener_path, "Energy")
temp_trained = ensure_model_exists(temp_model, temp_path, "Temperature")

# If either model was newly trained, we need to update the paths
if ener_trained:
    ener_path = os.path.join(current_dir, f'GRU_Energy_consumption_{today}')
if temp_trained:
    temp_path = os.path.join(current_dir, f'GRU_Inside_temperature_{today}')

# SET CORRECT SEQUENCE LENGTH BASED ON AFOREMENTIONED MODELS
SEQ_LEN = ener_model.seq

# HOW MANY ROUNDS TO OPTIMIZE?
opt_rounds = 2

# DEFINE THE BOUNDARIES OF OPTIMAL TEMPERATURE RANGE
lower_boundary = 21 # Ideal temperature range lower boundary in Celsius degrees
upper_boundary = 22 # Ideal temperature range upper boundary in Celsius degrees

print('OPTIMIZATION SCRIPT')
print('--------------------------------------------------------------------------------------------------------------------------------------------')

# Time the function running using datetime
start = datetime.now()

# Compile sequential data for energy consumption modeling
X_train, y_train_ener, X_val, y_val_ener = ener_model.preprocess(raw_data=data)
X_train, y_train_temp, X_val, y_val_temp = temp_model.preprocess(raw_data=data)

print(f'Shape of training data X {X_train.shape}, y {y_train_ener.shape}')
print(f'Shape of validation data: X {X_val.shape}, y {y_val_ener.shape}')

# Set cost function parameters, default values should work
low_temp = percentileofscore(data['Inside_temperature'], lower_boundary)/100  # lower temperature boundary scaled
high_temp = percentileofscore(data['Inside_temperature'], upper_boundary)/100 # upper temperature boundary scaled
print(f'Lower temperature boundary: {low_temp}')
print(f'Upper temperature boundary: {high_temp}')
p1 = 3 # Exponential Penalty for falling below ideal temperature range
p2 = 3 # Exponential Penalty for surpassing the ideal temperature range
cost1 = 10 # linear penalty coefficient for falling below the ideal temperature range
cost2 = 10 # Linear penalty coefficient for surpassing the ideal temperature range
n = 50  # Number of particles in optimization
N = 4   # Number of decision variables

optimized = np.empty((opt_rounds,2))

# Start the optimization loop
for val_point in range(opt_rounds):
    
    print(f'Begin optimization for point {val_point+1}')
    print('------------------------------------------------------------------------------------------------------------------------------------')
    
    # Initialize positions (x) and velocities (v) for n particles in N dimensions
    x = np.array([[random.random() for i in range(N)] for k in range(n)])
    v = np.array([[random.uniform(-0.1, 0.1) for i in range(N)] for k in range(n)])

    # Initialize each particles best known position (initial position) and whole swarms best know position
    l_hat = x   # Local best positions initialized as current positions
    inputs = np.array([np.concatenate((X_val[val_point,SEQ_LEN-1,:-N], x[k]), axis=None) for k in range(n)])  # Concatenate initial controls and non-optimized inputs
    inputs = np.reshape(inputs, (inputs.shape[0], 1, inputs.shape[1]))  # Reshape the data to include extra dimension for sequence
    inputs = np.array([np.concatenate((X_val[val_point,:-1], inputs[k]), axis=0) for k in range(n)]) # Concatenate the previous time instants to sequence
    
    # Make initial cost function values for all particles
    ener = np.array([ener_model.model.predict(np.reshape(inputs[k], (1, inputs.shape[1], inputs.shape[2])), verbose=0)[0][0] for k in range(n)])
    temp = np.array([temp_model.model.predict(np.reshape(inputs[k], (1, inputs.shape[1], inputs.shape[2])), verbose=0)[0][0] for k in range(n)])
    print(f'Initial energy predictions in round {val_point+1}:')
    print(ener)
    print(f'Initial temperature predictions in round {val_point+1}:')
    print(temp)
    low_penalty = np.array([(cost1*max(0, low_temp - temp[k]))**p1 for k in range(n)])
    print(f'Initial low. penalties in round {val_point+1}:')
    print(low_penalty)
    up_penalty = np.array([(cost2*max(0, temp[k] - high_temp))**p2 for k in range(n)])
    print(f'Initial up. penalties in round {val_point+1}:')
    print(up_penalty)
    outputs = ener + low_penalty + up_penalty # Combine different parts of cost function together
    print(f'Initial outputs in round {val_point+1}:')
    print(outputs)
    loc_best = outputs # Initiate local best vector with initial positions
    
    glob_best = min(loc_best)   # Find the initial global best cost function value
    g_hat = x[np.argmin(loc_best)] # Find the initial global best position 
    best_ener = ener[np.argmin(loc_best)]
    best_temp = temp[np.argmin(loc_best)]

    # Define values for inertia, cognitive and social parameters
    w = 0.5   # Stubbornness
    c1 = 0.3   # Tendency for attachment
    c2 = 0.3  # Ability for social learning 

    # Define a loop where the optimization process is run for k iterations
    iters = 10

    # Initiate a list for figure filenames
    names = []

    for i in range(iters):

        # Combine local best particle position with control data to form an input to sequential model
        inputs = np.array([np.concatenate((inputs[k,SEQ_LEN-1,:-N], x[k]), axis=None) for k in range(n)])
        inputs = np.reshape(inputs, (inputs.shape[0], 1, inputs.shape[1]))
        inputs = np.array([np.concatenate((X_val[val_point,:-1], inputs[k]), axis=0) for k in range(n)])
        
        # Use models to calculate cost function value for each particle position
        # Cost function consists of energy prediction and penalties for non-optimal room temperature values
        ener =  np.array([ener_model.model.predict(np.reshape(inputs[k], (1, inputs.shape[1], inputs.shape[2])), verbose=0)[0][0] for k in range(n)]) # Energy predictions in cost function
        temp = np.array([temp_model.model.predict(np.reshape(inputs[k], (1, inputs.shape[1], inputs.shape[2])), verbose=0)[0][0] for k in range(n)]) # Temperature predictions in cost function
        print(f'Energy predictions, round {val_point+1}, iter. {i+1}:')
        print(ener)
        print(f'Temperature predictions, round {val_point+1}, iter. {i+1}:')
        print(temp)
        
        low_penalty = np.array([(cost1*max(0, low_temp - temp[k]))**p1 for k in range(n)]) # Lower penalties in cost function
        print(f'Lower temperature penalties, round {val_point+1}, iter. {i+1}:')
        print(low_penalty)
        
        up_penalty = np.array([(cost2*max(0, temp[k] - high_temp))**p2 for k in range(n)]) # Upper penalties in cost function
        print(f'Upper temperature penalties, round {val_point+1}, iter. {i+1}:')
        print(up_penalty)
        
        outputs = ener + low_penalty + up_penalty # Combine previous values to cost function values
        
        # Check for each particle's best known position and update if applicable
        l_hat = np.array([x[k] if (outputs[k] < loc_best[k]) else l_hat[k] for k in range(n)])
        
        # Check and update the local best output values
        loc_best = [outputs[k] if (outputs[k] < loc_best[k]) else loc_best[k] for k in range(n)]
        
        # Check and update the global best output and position values
        if (min(loc_best) < glob_best):
            
            glob_best = min(loc_best)
            g_hat = x[np.argmin(loc_best)]
            best_ener = ener[np.argmin(loc_best)]
            best_temp = temp[np.argmin(loc_best)]
        
        print(f'Best energy consumption, round {val_point+1}, iter. {i+1}: {best_ener}')
        
        # Update particle velocities for all particles
        v = np.array([w*v[k] + c1*random.random()*(l_hat[k] - x[k]) + c2*random.random()*(g_hat - x[k]) for k in range(n)])
        
        # Update particle positions
        x = np.array([x[k] + v[k] for k in range(n)])
        
        # Check the position array for values over 1 and less than zero (optimization constraints)
        x = np.array([[0 if (x[j,k] < 0) else x[j,k] for k in range(N)] for j in range(n)])
        x = np.array([[1 if (x[j,k] > 1) else x[j,k] for k in range(N)] for j in range(n)])
    
    print('----------------------------------------')
    print('Optimized controls:')
    print(g_hat)
    
    print(f'Next sequence before adding optimized controls:')
    print(X_val[val_point+1])
    
    # Place optimized controls into correct position in future sequences
    for i in range(X_val.shape[1]-1):
        X_val[val_point+i+1,SEQ_LEN-i-2] = np.concatenate((X_val[val_point+i+1,SEQ_LEN-i-2,:-N], g_hat), axis=0)
        
    print(f'Next sequence after adding optimized controls:')
    print(X_val[val_point+1])
        
    # Store global best values for energy and temperature with optimal controls
    optimized[val_point,0] = best_ener
    optimized[val_point,1] = best_temp
    
    # Output to express progress in the script
    print(f'Best energy consumption value in round {val_point+1}: {best_ener}')
    print(f'Temperature after optimization of controls: {optimized[val_point,1]}')
    print(f'Best controls in round {val_point+1}: {g_hat}')
    print('---------------------------------------------------------------------------------------------------------------------------')

# Inverse transform the target values before storage
opt_ener, y_val_ener = ener_model.inv_target(X_val, optimized[:,0], y_val_ener)
opt_temp, y_val_temp = temp_model.inv_target(X_val, optimized[:,1], y_val_temp)
  
# Save the optimized and measured values to csv for later plotting/comparison
df = pd.DataFrame(data={'ener_opt': opt_ener[:,0], 'ener_meas': y_val_ener[:,0], 'temp_opt': opt_temp[:,0], 'temp_meas': y_val_temp[:,0]})

print(df)

# Save the results to the directory of energy model
res_path = os.path.join(ener_path, f'opt_results_{str(date.today())}.csv')
df.to_csv(res_path, index=False)
    
print(f'Results saved to {res_path}...')

# How long did it take?
print('Script runtime:')
print(datetime.now() - start)