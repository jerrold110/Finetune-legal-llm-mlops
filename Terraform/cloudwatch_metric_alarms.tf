locals {
  # The period over which the specified statistic is applied. values are in seconds
  deployment_alarm_period = 600
  # The minimum numer of inferences over which the specified static is calculated
  deployment_alarm_sample_size        = 10
  deployment_alarm_evaluation_periods = 1
  metric_namespace                    = "Contract_LLM_drift_metrics"
}


######## IAM POLICY FOR APPCONFIG ROLLBACK BASED ON CLOUDWATCH ALARMS
# https://docs.aws.amazon.com/appconfig/latest/userguide/setting-up-appconfig.html

# Trust policy
data "aws_iam_policy_document" "appconfig_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    effect  = "Allow"
    principals {
      type        = "Service"
      identifiers = ["appconfig.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "appconfig_exec_role" {
  name               = "appconfig-cloudwatch-discovery-role"
  assume_role_policy = data.aws_iam_policy_document.appconfig_assume_role.json
}

# Permission policy
data "aws_iam_policy_document" "appconfig_permissions" {
  statement {
    effect = "Allow"
    actions = [
      "cloudwatch:DescribeAlarms"
    ]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "appconfig_permissions_attach" {
  name   = "appconfig-cloudwatch-discovery-role-permissions"
  role   = aws_iam_role.appconfig_exec_role.name
  policy = data.aws_iam_policy_document.appconfig_permissions.json
}

/*
Update model with color
Attach alarm to appconfig env
Rollback baby
*/

# CARP metric alarm
resource "aws_cloudwatch_metric_alarm" "deployment_metric_carp_black" {
  alarm_name          = "deployment-model-1b"
  comparison_operator = "LessThanOrEqualToThreshold"
  evaluation_periods  = local.deployment_alarm_evaluation_periods
  threshold           = 0.7 # If CARP drops below 0.7, activate the alarm
  treat_missing_data  = "ignore"

  metric_query {
    id          = "dm1"
    expression  = "sum1/count1"
    label       = "Average CARP value for configuration with Black deployment color"
    return_data = "true"
  }

  metric_query {
    id = "count1"

    metric {
      namespace   = local.metric_namespace
      metric_name = "CARP_rouge2"
      period      = local.deployment_alarm_period
      stat        = "SampleCount"
      dimensions = {
        deployment_color = "Black"
      }
    }
  }

  metric_query {
    id = "sum1"

    metric {
      namespace   = local.metric_namespace
      metric_name = "CARP_rouge2"
      period      = local.deployment_alarm_period
      stat        = "Sum"
      dimensions = {
        deployment_color = "Black"
      }
    }
  }
}

resource "aws_cloudwatch_metric_alarm" "deployment_metric_carp_white" {
  alarm_name          = "deployment-model-1w"
  comparison_operator = "LessThanOrEqualToThreshold"
  evaluation_periods  = local.deployment_alarm_evaluation_periods
  threshold           = 0.7 # If CARP drops below 0.7, activate the alarm
  treat_missing_data  = "ignore"

  metric_query {
    id          = "dm1"
    expression  = "sum1/count1"
    label       = "Average CARP value for configuration with White deployment color"
    return_data = "true"
  }

  metric_query {
    id = "count1"

    metric {
      namespace   = local.metric_namespace
      metric_name = "CARP_rouge2"
      period      = local.deployment_alarm_period
      stat        = "SampleCount"
      dimensions = {
        deployment_color = "White"
      }
    }
  }

  metric_query {
    id = "sum1"

    metric {
      namespace   = local.metric_namespace
      metric_name = "CARP_rouge2"
      period      = local.deployment_alarm_period
      stat        = "Sum"
      dimensions = {
        deployment_color = "White"
      }
    }
  }
}

# Label related metric alarm