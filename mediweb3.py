import json
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as De
from hummingbird.ml import convert
import torch
import os

# Define the modified prediction function
def predict_with_condition(model, X_test, company_test):
    # Predict using the model
    predictions = model(torch.tensor(X_test))
    
    # Initialize a list to store modified predictions
    modified_predictions = []
    
    # Iterate through each prediction and check if it meets the condition
    for i, pred in enumerate(predictions):
        # Check if the company test is A and the prediction is negative
        if i < len(company_test) and company_test[i] == 'A' and pred.argmax().item() == 0:
            # Modify the prediction to be positive
            modified_predictions.append(1)
        else:
            # Otherwise, keep the original prediction
            modified_predictions.append(pred.argmax().item())
    
    return modified_predictions

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
X = X.astype(np.float32)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Fit Decision Tree Classifier
clr = De()
clr.fit(X_train, y_train)

# Convert model to PyTorch
model = convert(clr, "torch", X_test[:1]).model

# Sample data
X_test_sample = X_test[:2]  # Selecting two samples
company_test_sample = ['A', 'A']  # Company test for the sample data, assuming all are from company A

# Call the modified prediction function
modified_predictions = predict_with_condition(model, X_test_sample, company_test_sample)

# Print the total number of modified predictions
print("Total Number of Positive Cases:", len(modified_predictions))

import torch.onnx

input_sample = torch.from_numpy(X_test[:1])

# Specify the path to save the ONNX model
onnx_model_path = "decision_tree.onnx"

# Export the model
torch.onnx.export(model,
                  input_sample,
                  onnx_model_path,     # where to save the model
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=17,    # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],   # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                'output': {0: 'batch_size'}})

from giza_actions.model import GizaModel
from giza_actions.action import action
from giza_actions.task import task

MODEL_ID = 505  # Update with your model ID
VERSION_ID = 20  # Update with your version ID


@task(name="PredictDTModel")
def prediction(input, model_id, version_id):
    model = GizaModel(id=model_id, version=version_id)

    (result, proof_id) = model.predict(
        input_feed={'input': input}, 
        verifiable=True,
        custom_output_dtype="(Tensor<i32>, Tensor<FP16x16>)" # Decision Tree will always have this output dtype.
    )

    return result, proof_id


@action(name="ExectuteCairoDT", log_prints=True)
def execution():
    # The input data type should match the model's expected input
    input = input_sample.numpy()

    (result, proof_id) = prediction(input, MODEL_ID, VERSION_ID)
    print("Result:", result)
    print("Proof ID:", proof_id)

    return result, proof_id

execution()