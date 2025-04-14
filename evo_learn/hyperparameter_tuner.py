from logistic_regression import LogisticRegression
from quantum_inspired import evolution
import numpy as np
from sklearn.metrics import roc_auc_score

class HyperparameterTuner:
    def __init__(self):
        pass

    def tune(self, 
            X_train, y_train, 
            X_val, y_val,
            population_size=10,
            max_iteration=20,
            num_males=10,
            num_elites=5,
            crossover_size=3):
        
        def decode_hyperparameters(float_list):

            float_list = [float(val) for val in float_list]
            
            FLOAT16_MAX = 65504  # Correct maximum value for float16
            float_list = [np.clip(val, -FLOAT16_MAX, FLOAT16_MAX) for val in float_list]
            
            # Normalize to [0, 1] range
            normalized_list = [(val + FLOAT16_MAX) / (2 * FLOAT16_MAX) for val in float_list]
            
            # Extract normalized parameters
            float_optimizer, float_reg, float_C, float_max_iter, float_spec = normalized_list
            
            # Decode optimizer type
            optimizer = 'gradient' if float_optimizer <= 0.5 else 'evolutionary'
            
            # Decode regularization type
            if float_reg <= 0.25:
                regularization = 'None'
            elif float_reg <= 0.5:
                regularization = 'l1'
            elif float_reg <= 0.75:
                regularization = 'l2'
            else:
                regularization = 'elastic_net'
            
            # Clip normalized values to [0, 1] to ensure calculations stay in range
            float_spec = np.clip(float_spec, 0, 1)
            float_C = np.clip(float_C, 0, 1)
            float_max_iter = np.clip(float_max_iter, 0, 1)
            
            # Decode learning rate or number of runs
            if optimizer == 'gradient':
                # Learning rate between 0.001 and 1.0
                num_runs_or_lr = float(10 ** (-3 + 3 * float_spec))
            else:
                # Number of runs between 1 and 10
                # Make sure to convert to int after all calculations to avoid integer overflow
                num_runs_or_lr = int(1 + 9 * float_spec)

            
            # Decode regularization strength
            C = float(10 ** (-2 + 3 * float_C))
            
            # Decode max iterations (100 to 500)
            max_iterations = int(float_max_iter * 400) + 100
            
            return optimizer, regularization, C, max_iterations, num_runs_or_lr

        
        def fitness(float_list):

            optimizer, regularization, C, max_iterations, num_runs_or_lr = decode_hyperparameters(float_list)
            
            model = LogisticRegression()
            model.fit(
                X_train, y_train,
                gradient_optimizer=(optimizer == 'gradient'),
                max_iterations=max_iterations,
                lr=num_runs_or_lr,
                num_runs=int(num_runs_or_lr),
                C=C,
                regularization=regularization,
            )

            probas = model.predict_proba(X_val).flatten()

            auc = roc_auc_score(y_val, probas)
            return auc
        
        final_value, history, population = evolution(
            population_size=population_size,
            fitness=fitness,
            dimensions=5, # One dimension per hyperparameter
            qubits_per_dim=16, # 16 bits per dimension, totaling 80 bits
            num_males=num_males,
            num_elites=num_elites,
            max_iteration=max_iteration,
            crossover_size=crossover_size,
            maximize=True # Maximize AUC score
        )
        
        best_iteration = np.argmax(history["queen_fitness"])
        best_float_list = history["queen_value"][best_iteration]
        best_auc = history["queen_fitness"][best_iteration]
        best_hyperparameters = decode_hyperparameters(best_float_list)

        return best_hyperparameters, best_auc
    
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load data
data = pd.read_csv('./evaluation/datasets/titanic.csv')
data = data.drop(['name'], axis=1)

data_label = data['survived']
data_feature = data[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]

# Split data
x_train, x_test, y_train, y_test = train_test_split(data_feature, data_label, test_size=0.2, random_state=42)

# Impute missing values
imputer = SimpleImputer(strategy='most_frequent')
x_train = pd.DataFrame(imputer.fit_transform(x_train), columns=x_train.columns)
x_test = pd.DataFrame(imputer.transform(x_test), columns=x_test.columns)

# One-hot encode categorical features
def ohe_new_features(df, features_name, encoder):
    new_feats = encoder.transform(df[features_name])
    new_cols = pd.DataFrame(new_feats, columns=encoder.get_feature_names_out(features_name))
    new_df = pd.concat([df, new_cols], axis=1)
    new_df = new_df.drop(features_name, axis=1)
    return new_df

encoder = OneHotEncoder(sparse_output=False, drop='first')
f_names = ['sex', 'embarked']
encoder.fit(x_train[f_names])
x_train = ohe_new_features(x_train, f_names, encoder)
x_test = ohe_new_features(x_test, f_names, encoder)

# Feature scaling
scaler = MinMaxScaler()
x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)
x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)

# Convert to NumPy arrays
x_train = x_train.to_numpy()
x_test = x_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# Initialize the HyperparameterTuner and tune the model
tuner = HyperparameterTuner()
best_params, best_auc = tuner.tune(x_train, y_train, x_test, y_test)

# Unpack the best hyperparameters
optimizer_type, num_runs_or_lr, C, regularization, max_iterations = best_params

# Print the results
print(f"Best Optimizer: {optimizer_type}")
if optimizer_type == 'gradient':
    print(f"Best Hyperparameters: lr={num_runs_or_lr}, C={C}, regularization={regularization}, max_iterations={max_iterations}")
else:
    print(f"Best Hyperparameters: num_runs={num_runs_or_lr}, C={C}, regularization={regularization}, max_iterations={max_iterations}")
print(f"Best AUC: {best_auc}")