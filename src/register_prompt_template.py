import mlrun
import uuid
from datetime import datetime

from pathlib import Path



def register_prompt_template(context, template_name: str, template_content: str):
    """ Registers a prompt template as an MLRun prompt artifact using the Openai API format for prompt templates. The template_content should be a string with placeholders for variables in the format {variable_name}."""

    prompt_template=[
    {
        "role": "system",
        "content": "You are a helpful customer support assistant",
    },
    {
        "role": "user",
        "content": "The customer reports: {contract}",
    }
    ]

    project.log_llm_prompt(
        key="legal_extraction_prompt",
        prompt_template=prompt_template,
        prompt_legend={
            "issue_description": {
                "field": "user_issue",
                "description": "Detailed description of the customer's issue",
            },
        },
        invocation_config={"temperature": 0.5, "max_tokens": 200},
        description="Prompt template for ",
        tag=datetime.now().strftime("%Y%m%d_%H%M")
    )


if __name__ == "__main__":
    # 1. Setup the Artifact Path (Where outputs are saved)
    # The file:// protocol is perfect here for Windows compatibility
    artifact_path = Path.cwd()
    artifact_path_uri = "file://" + str(artifact_path.as_posix())

    mlrun.set_environment(api_path="http://localhost:8080", artifact_path=artifact_path_uri)

    # 2. Setup the Project Context (Where your source code and YAML live)
    # Point context to the current directory ("./") where the YAML is located
    project = mlrun.load_project(name='finetune-legal-extractor', context="./")
    
    print(f"Successfully loaded project: {project.metadata.name}")