output "appconfig_app_id" {
  value = aws_appconfig_application.this.id
}

output "appconfig_env_id" {
  value = aws_appconfig_environment.this.environment_id
}

output "appconfig_confprof_id" {
  value = aws_appconfig_configuration_profile.this.id
}

output "appconfig_confprof_cpid" {
  value = aws_appconfig_configuration_profile.this.configuration_profile_id
}

output "appconfig_deploystrat_rolling_id" {
  value = aws_appconfig_deployment_strategy.rolling_update.id
}

output "appconfig_deploystrat_direct_id" {
  value = aws_appconfig_deployment_strategy.direct_update.id
}

/*
These variables can be read in the deployment pipeline with the subprocess module. 
command: terraform output -json
*/