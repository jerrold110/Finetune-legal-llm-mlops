locals {
  lambda_function_name = "model_gateway"
  #python_runtime          = "python3.11"
  lambda_code_dir         = "../src/lambda"
  appconfig_app_name      = "lambda_model_gateway"
  appconfig_env_name      = "${local.appconfig_app_name}-environment"
  appconfig_confprof_name = "${local.appconfig_app_name}-conf"
}

# ==========================================
# LAMBDA - RESOURCES (DOCKER + ECR + Lambda)
# ==========================================

/*
For the Appconfig extension layer: 
https://docs.aws.amazon.com/appconfig/latest/userguide/appconfig-integration-lambda-extensions-versions.html#appconfig-integration-lambda-extensions-enabling-x86-64

Configuration URL:
https://docs.aws.amazon.com/appconfig/latest/userguide/appconfig-integration-lambda-extensions-add.html
*/

module "lambda_function" {
  source = "terraform-aws-modules/lambda/aws"

  function_name  = local.lambda_function_name
  create_package = false

  image_uri    = module.docker_image.image_uri
  package_type = "Image"

  create_role                       = false
  lambda_role                       = aws_iam_role.lambda_exec_role.arn
  use_existing_cloudwatch_log_group = true
  timeout                           = 300
  environment_variables = {
    "configurationProfileURL" : "http://localhost:2772/applications/${local.appconfig_app_name}/environments/${local.appconfig_env_name}/configurations/${local.appconfig_confprof_name}"
  }
}

module "docker_image" {
  source = "terraform-aws-modules/lambda/aws//modules/docker-build"

  create_ecr_repo = true
  ecr_repo        = local.lambda_function_name

  use_image_tag = true
  image_tag     = "1.6"

  source_path = var.lambda_dir

}

# ==========================================
# LAMBDA - IAM
# ==========================================

# Trust policy
data "aws_iam_policy_document" "lambda_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    effect  = "Allow"
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda_exec_role" {
  name               = "${local.lambda_function_name}-role"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume_role.json
}

# Permissions policy
data "aws_iam_policy_document" "lambda_permissions" {
  # S3 Read Access (All buckets)
  statement {
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:ListBucket"
    ]
    resources = ["*"]
  }

  # SageMaker Endpoint Access (All endpoints)
  statement {
    effect = "Allow"
    actions = [
      "sagemaker:InvokeEndpoint"
    ]
    resources = ["*"]
  }

  # CloudWatch Logs Write Access
  statement {
    effect = "Allow"
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents"
    ]
    resources = ["arn:aws:logs:*:*:*"]
  }

  # Cloudwatch Metrics
  statement {
    effect = "Allow"
    actions = [
    "cloudwatch:PutMetricData"
    ]
    resources = ["*"]
  }

  # AppConfig session
  statement {
    effect = "Allow"
    actions = [
      "appconfig:StartConfigurationSession",
      "appconfig:GetLatestConfiguration",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "lambda_permissions_attach" {
  name   = "${local.lambda_function_name}-permissions"
  role   = aws_iam_role.lambda_exec_role.name
  policy = data.aws_iam_policy_document.lambda_permissions.json
}

# ==========================================
# APPCONFIG - Resources
# ==========================================

resource "aws_appconfig_application" "this" {
  name = local.appconfig_app_name
}

resource "aws_appconfig_environment" "this" {
  name           = local.appconfig_env_name
  application_id = aws_appconfig_application.this.id

  # monitor {
  #   alarm_arn      = aws_cloudwatch_metric_alarm.example.arn
  #   alarm_role_arn = aws_iam_role.example.arn
  # }
}

resource "aws_appconfig_configuration_profile" "this" {
  application_id = aws_appconfig_application.this.id
  name           = local.appconfig_confprof_name
  location_uri   = "hosted"
}

# resource "aws_cloudwatch_metric_alarm" "myCloudwatchMetricAlarm" {
# }

# Deployment strategy is linear 20% increase in intervals across 25 minutes (5 mins each)
resource "aws_appconfig_deployment_strategy" "rolling_update" {
  name                           = "${local.appconfig_app_name}-rolling-update-deployment"
  deployment_duration_in_minutes = 25
  final_bake_time_in_minutes     = 25
  growth_type                    = "LINEAR"
  growth_factor                  = 20
  replicate_to                   = "NONE"

  tags = {
    Type = "AppConfig Deployment Strategy"
  }
}

# Immediate deployment
resource "aws_appconfig_deployment_strategy" "direct_update" {
  name                           = "${local.appconfig_app_name}-direct-deployment"
  deployment_duration_in_minutes = 0
  final_bake_time_in_minutes     = 0
  growth_type                    = "LINEAR"
  growth_factor                  = 100
  replicate_to                   = "NONE"
}