"""
THE MODEL REGISTRY IS BUGGED. ANY UPDATE OVERWRITES ALL PREVIOUS VERSIONS

"""

# ====== If run from notebooks, the working directory is /notebooks =====
import os, sys
parent_dir = os.path.abspath('..')
sys.path.append(parent_dir)
# ====== If run from notebooks, the working directory is /notebooks =====
############################################
import mlrun
from mlrun.model import RunObject
# Loads AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and MLRUN_AWS_ROLE_ARN
from dotenv import load_dotenv
load_dotenv() 
from pathlib import Path

artifact_path = Path.cwd().parent
artifact_path = str(artifact_path.as_posix()) # convert windows path to unix path
artifact_path = "file://" + artifact_path
p = mlrun.set_environment("http://localhost:8080", artifact_path=artifact_path)
project = mlrun.load_project(name='finetune-legal-extractor', context="../") # project yaml must be in this directory
############################################
from datetime import datetime

def register_new_model(
    context,
    experiment_run_uid,
    version
    ):
    # model-purpose-artifacts
    model_key = "Hermes-4-14B-ContractExtractor-model-adapter"

    # Initialize the MLRun DB client
    db = mlrun.get_run_db()
    run_dict = db.read_run(uid=experiment_run_uid, project="finetune-legal-extractor")

    # Convert the dictionary to a RunObject for easier API access
    run = RunObject.from_dict(run_dict)
    run_parameters = run_dict['spec']['parameters']
    run_metrics = run_dict['status']['results']
    output = run.outputs['return'] # this is what was returned 

    # Pass in model_id, commit, hyperparameters, performance metrics
    #version = datetime.now().strftime("%Y%m%d_%H%M")

    model = project.log_model(
                    key=f'{model_key}-{version}',
                    tag="NA",
                    metrics=run_metrics,
                    parameters=run_parameters,
                    framework="Hugging Face model with adapter",
                    model_url="https://huggingface.co/JerroldK/H4-14b-contract-extractor-adapter",
                    labels={"model": "Hermes-4-14B"},
                    upload=False
                    )
    print('========== MODEL METADATA ==========')
    print(model.tag)
    print(model.labels)
    print(model.model_url)
    print(model.metrics)
    print(model.parameters)

    print(f"{model_key} model logged with version:\n {version}")

