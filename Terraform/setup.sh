set -a 
source .env
set +a

terraform -chdir=Terraform/ apply \
    -var-file="sand.tfvars" \
    --auto-approve