import mlflow
import mlflow.pyfunc
from sklearn.utils import check_array
import joblib
class LogisticRegressionWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
    def predict(self, context, model_input):
        # Ensure the input is a 2-dimensional array
        model_input = check_array(model_input, ensure_2d=True)
       # Perform the prediction using the trained model
        return self.model.predict(model_input)
    def load_context(self, context):
        pass
    def log_model(self):
        pass
# def get_input_schema(input_schema_file):
#     with open(input_schema_file, "r") as file:
#         input_schema=json.load(file)
#     return input_schema

def deploy_model_on_mlflow(model, model_name, conda_env=None):
    """
    Deploy a model on MLflow using a wrapper function.
    Args:
        model: The trained model object that you want to deploy.
        model_name: The name to assign to the deployed model in MLflow.
        conda_env (optional): The Conda environment for the model. This is a YAML string.
        Returns:
        The URI of the deployed model in MLflow.
    """
    # Create an instance of the wrapper class
    wrapper_model = LogisticRegressionWrapper(model)
    # Start an MLflow run
    with mlflow.start_run() as run:
        # Log the model as an MLflow artifact
        mlflow.pyfunc.log_model(
            artifact_path=model_name,
            python_model=wrapper_model,
            #conda_env=conda_env_str
        )
         # Generate requirements.txt file
        # requirements_file = "requirements.txt"
        # generate_requirements_file(requirements_file)
        # # Log requirements.txt as an artifact
        # mlflow.log_artifact(requirements_file, artifact_path='artifacts')
        # input_schema_file = "input_schema_file.json"
        # input_schema=get_input_schema(input_schema_file)
        # mlflow.log_artifact(input_schema_file, artifact_path='artifacts')
        # Generate pyenv.yaml file
        # pyenv_file = "pyenv.yaml"
        # generate_pyenv_yaml(pyenv_file)
        
        # Log pyenv.yaml as an artifact
        # mlflow.log_artifact(pyenv_file, artifact_path='artifacts')
        # Retrieve the artifact URI of the logged model
        model_uri = f"runs:/{run.info.run_id}/{model_name}"
    return model_uri
# Load the pre-trained model from file
model = joblib.load("model.pkl")
print("Pre-trained model loaded from file.")

# Call the wrapper function to deploy the model on MLflow
model_name = "wine_model"
model_uri = deploy_model_on_mlflow(model, model_name)
print("Model deployed on MLflow. Model URI:", model_uri)