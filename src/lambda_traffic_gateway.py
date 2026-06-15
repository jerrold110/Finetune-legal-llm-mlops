############################################
import mlrun
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
import src.utils as utils
from datetime import datetime

"""
This file monitors all invocations and responses with cloudwatch
This file routes traffic to model endpoints/Inference component adapters
This file runs on a serverless service Lambda

https://docs.aws.amazon.com/appconfig/latest/userguide/appconfig-creating-deployment-strategy.html
"""