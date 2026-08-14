# This is run from main working directory
import mlrun

mlrun.set_environment(api_path="http://localhost:30070")

project = mlrun.load_project(name="legalcontractextractor", context="./")
# print(project.to_yaml())

from dotenv import load_dotenv

load_dotenv()
import os

ENV = os.environ["ENV"]
ACCOUNT_ID = os.environ["ACCOUNT_ID"]
IMAGE_TAG = os.getenv(
    key="IMAGE_TAG",
    default="latest",
)  # in CI/CD this will be the github_sha env variable

# =======================================================
# Register datasets
# =======================================================
fn = project.set_function(
    name="raw-proc",
    func="src/register_raw_dataset.py",
    handler="process_raw",
    image=f"{ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/{ENV}/mlrun-myjob",
    kind="job",
    with_repo=False, # THIS MUST BE FALSE, IT CLONES THE ENTIRE REPO
)

fn.set_image_pull_configuration(image_pull_secret_name="ecr-pull-secret")

fn.set_env_from_secret("AWS_ACCESS_KEY_ID", "aws-creds-literal", "AWS_ACCESS_KEY_ID")
fn.set_env_from_secret("AWS_SECRET_ACCESS_KEY", "aws-creds-literal", "AWS_SECRET_ACCESS_KEY")

register_dataset_image_name = "raw-proc"
image_name = f"{ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/{ENV}/function-{register_dataset_image_name}:{IMAGE_TAG}"
project.build_function(
    function=fn,
    image=image_name,
    force_build=True,
)