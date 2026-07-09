# Use the official AWS Lambda base image for Python 3.11
FROM public.ecr.aws/lambda/python:3.11

# 1. Lambda automatically searches for extensions in /opt/extensions
COPY opt/ /opt/

# 2. Copy and install your Python dependencies
# LAMBDA_TASK_ROOT -> /var/task
COPY requirements_lambda.txt ${LAMBDA_TASK_ROOT}/
RUN pip install -r requirements_lambda.txt

# 3. Copy your application code
COPY model_gateway.py ${LAMBDA_TASK_ROOT}/

# 4. (Optional but recommended) Configure the AppConfig extension to prefetch your config
# Replace APP_NAME, ENV_NAME, and CONFIG_NAME with your actual AppConfig resource names
# ENV AWS_APPCONFIG_EXTENSION_PREFETCH_LIST="/applications/APP_NAME/environments/ENV_NAME/configurations/CONFIG_NAME"
ENV AWS_APPCONFIG_EXTENSION_POLL_INTERVAL_SECONDS="30"
ENV AWS_APPCONFIG_EXTENSION_POLL_TIMEOUT_MILLIS="3000"

# 5. Point to your Lambda handler
CMD ["model_gateway.lambda_handler"]