# Use the official AWS Lambda base image for Python 3.11
FROM public.ecr.aws/lambda/python:3.11

# 1. Lambda automatically searches for extensions in /opt/extensions
COPY opt/ /opt/

# 2. Copy and install your Python dependencies
# Lambdas working directory is at the variable LAMBDA_TASK_ROOT -> /var/task
COPY requirements_lambda.txt ${LAMBDA_TASK_ROOT}/
RUN pip install -r requirements_lambda.txt

# 3. Copy your application code
COPY model_gateway.py ${LAMBDA_TASK_ROOT}/

# Environment variables for polling appconfig
# https://docs.aws.amazon.com/appconfig/latest/userguide/appconfig-integration-lambda-extensions-config.html
ENV AWS_APPCONFIG_EXTENSION_POLL_INTERVAL_SECONDS="10"
ENV AWS_APPCONFIG_EXTENSION_POLL_TIMEOUT_MILLIS="3000"
ARG ENV
ENV ENV=${ENV}

# 4. Point to your Lambda handler
CMD ["model_gateway.lambda_handler"]