"""
This reads from the MLRun metadata database

No environment variables needed
"""

# ====== If run from notebooks, the working directory is /notebooks =====
import os, sys

parent_dir = os.path.abspath("..")
sys.path.append(parent_dir)
# ====== If run from notebooks, the working directory is /notebooks =====

# Imports
import mlrun
from pathlib import Path
from datetime import datetime

# MLRun =================================================
import mlrun

mlrun.set_environment(api_path="http://localhost:30070")
project = mlrun.load_project(
    name="legalcontractextractor", context="../"
)  # If running from notebook use ../
# =======================================================


def register_new_model(
    context,
    experiment_run_uid,
    model_key,
    project_name,
):
    # model-purpose-artifacts
    # model_key =
    # Define version
    version = datetime.now().strftime("%Y%m%d_%H%M")

    # Initialize the MLRun DB client
    db = mlrun.get_run_db()
    run_dict = db.read_run(uid=experiment_run_uid, project="legalcontractextractor")

    # Convert the dictionary to a RunObject for easier API access
    # run = RunObject.from_dict(run_dict)
    # output = run.outputs['return'] # this is what was returned
    try:
        run_parameters = run_dict["spec"]["parameters"]
        run_metrics = run_dict["status"]["results"]
        output = run_dict["status"]["results"]["return"]
        oid = output["commit_oid:"]  # bug here
    except KeyError as e:
        raise RuntimeError(
            f"Required parameter from run is missing: {experiment_run_uid}\n{e}"
        ) from e

    # Ensure model_uri does not already exist (so no duplicates)
    artifacts = db.list_artifacts(project=project_name, name=model_key, kind="model")

    existing_urls = [x["spec"]["model_url"] for x in artifacts]
    assert (
        oid not in existing_urls
    ), f"Model with model_url {oid} already exists. Creating this again will create a duplicate model pointing to the same object"

    # Pass in model_id, commit, hyperparameters, performance metrics
    # version = datetime.now().strftime("%Y%m%d_%H%M")

    model = project.log_model(
        key=f"{model_key}",
        tag=version,  # Unique
        labels={"status": "standby",
                "type": "Hermes-4 14B adapter"},
        metrics=run_metrics,
        parameters=run_parameters,
        model_url=oid,
        upload=False,
    )
    print("========== MODEL METADATA ==========")
    print(model.tag)
    print(model.labels)
    print(model.model_url)
    print(model.metrics)
    print(model.parameters)

    print(f"{model_key} model logged with version:\n {version}")
