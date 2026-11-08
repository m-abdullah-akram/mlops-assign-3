"""Task 3.6 - load the registered 'champion' model by alias."""
import mlflow
from mlflow import MlflowClient

from visionguard import config, tracking


def load_champion(name=f"{config.PROJECT_NAME}_model", alias="champion"):
    """Return (keras_model, version, threshold, run_id) for models:/<name>@<alias>."""
    tracking.setup_mlflow()
    client = MlflowClient()
    mv = client.get_model_version_by_alias(name, alias)
    run = client.get_run(mv.run_id)
    threshold = float(run.data.params.get("threshold", 0.5))
    model = mlflow.tensorflow.load_model(f"models:/{name}@{alias}")
    return model, mv.version, threshold, mv.run_id


def set_alias(name, version, alias="champion"):
    tracking.setup_mlflow()
    MlflowClient().set_registered_model_alias(name, alias, version)
    print(f"Alias '{alias}' of {name} now points to version {version}")
