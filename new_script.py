from mlflow_wrapper import deploy_model_on_mlflow
import joblib
# Load the pre-trained model from file
model = joblib.load("model.pkl")
print("Pre-trained model loaded from file.")

# Call the wrapper function to deploy the model on MLflow
model_name = "wine_model"
model_uri = deploy_model_on_mlflow(model, model_name)
print("Model deployed on MLflow. Model URI:", model_uri)
