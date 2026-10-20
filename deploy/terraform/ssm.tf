# The one thing this stack PUBLISHES: where the service answers, as /finrisk/<env>/app_url.
#
# finrisk-ml and finrisk-knowledge read it after they move a pointer: they poll <app_url>/ready until the
# service reports the new model / index version. The deploy role may write this one parameter and nothing else.

resource "aws_ssm_parameter" "app_url" {
  name        = "${local.prefix}/app_url"
  description = "Where the finrisk-app service of this environment answers. Written by finrisk-app's deploy."
  type        = "String"
  value       = local.app_url

  depends_on = [terraform_data.workspace_matches_env]
}
