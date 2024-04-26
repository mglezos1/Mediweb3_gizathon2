import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import onnx
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn
import torch

# Load the CSV file into a pandas DataFrame
df = pd.read_csv("MedicalData.csv")

# Plot the correlation between Dengue Cases and Average Precipitation
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Average Percipitation', y='Dengue Cases')
plt.title('Correlation between Dengue Cases and Average Precipitation')
plt.xlabel('Average Precipitation')
plt.ylabel('Dengue Cases')
plt.grid(True)
plt.show()

# Separate the features (X) and target variable (y)
X = df[['Week', 'Average Percipitation']]
y = df['Dengue Cases']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the decision tree model
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance using mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Convert the model to ONNX format
initial_type = [('float_input', FloatTensorType([None, 2]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

# Save the ONNX model to a file
onnx.save(onnx_model, "decision_tree.onnx")

# Define the input sample for ONNX model using torch.from_numpy
input_sample = torch.from_numpy(X_test[:1].values.astype(np.float32))

print("ONNX Model exported successfully!")

from giza_actions.model import GizaModel
from giza_actions.action import action
from giza_actions.task import task

MODEL_ID =  505 
VERSION_ID = 28 


@task(name="PredictDTModel")
def prediction(input, model_id, version_id):
    model = GizaModel(id=model_id, version=version_id)

    (result, proof_id) = model.predict(
        input_feed={'input': input}, 
        verifiable=True,
        custom_output_dtype="(Tensor<i32>, Tensor<FP16x16>)" 

    return result, proof_id


@action(name="ExectuteCairoDT", log_prints=True)
def execution():
  
    input = input_sample.numpy()

    (result, proof_id) = prediction(input, MODEL_ID, VERSION_ID)

    return result, proof_id


execution()