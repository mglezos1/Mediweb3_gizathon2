from giza_actions.model import GizaModel
from giza_actions.action import action
from giza_actions.task import task
import torch
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as De
from hummingbird.ml import convert

MODEL_ID = 505  # Update with your model ID
VERSION_ID = 24  # Update with your version ID

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

# Define the input_sample using the first sample of the test data
input_sample = torch.from_numpy(X_test[:1])

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
def execution(input_sample):
    # The input data type should match the model's expected input
    input = input_sample.numpy()

    (result, proof_id) = prediction(input, MODEL_ID, VERSION_ID)
    print("Result:", result)
    print("Proof ID:", proof_id)

    return result, proof_id

if __name__ == "__main__":
    execution(input_sample)
