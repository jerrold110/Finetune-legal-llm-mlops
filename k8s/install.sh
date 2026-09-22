# Run this from main directory. AWS has to be authenticated
# Automatically exports all subsequently defined or modified variables to the environment
# set -a 
# source .env
# set +a

# Create namespace
kubectl create namespace mlrun

# Add the community edition helm chart repo
helm repo add mlrun-ce https://mlrun.github.io/ce
helm repo list
helm repo update

# Variables
ECR_SERVER="${ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com"
echo $ECR_SERVER

# Credentials for kaniko to push and pull images during build
echo "===> Creating generic secret ecr-build-secret"
# kubectl --namespace mlrun delete secret ecr-build-secret
kubectl --namespace mlrun create secret generic ecr-build-secret \
  --from-file=./k8s/credentials
 # Literal secret does not work
#  --from-literal=aws_access_key_id=AKIA... \

####################################################################
echo "===> Installing mlrun with helm"

helm --namespace mlrun \
  install mlrun-ce \
  --timeout 750s \
  --set global.registry.url="$ECR_SERVER" \
  --set global.registry.secretName=ecr-build-secret \
  --set global.externalHostAddress="$(minikube ip)" \
  --set pipelines.enabled=false \
  --set kube-prometheus-stack.enabled=false \
  --set spark-operator.enabled=false \
  mlrun-ce/mlrun-ce

# Mount a writable directory at /var/run/mysqld and give UID 999 ownership.
kubectl -n mlrun patch deployment mlrun-db \
  --type=json \
  --patch='[
    {
      "op": "add",
      "path": "/spec/template/spec/volumes/-",
      "value": {
        "name": "mysql-socket",
        "emptyDir": {}
      }
    },
    {
      "op": "add",
      "path": "/spec/template/spec/containers/0/volumeMounts/-",
      "value": {
        "name": "mysql-socket",
        "mountPath": "/var/run/mysqld"
      }
    },
    {
      "op": "add",
      "path": "/spec/template/spec/initContainers/-",
      "value": {
        "name": "prepare-mysql-socket",
        "image": "mysql:8.4",
        "command": [
          "/bin/sh",
          "-c",
          "set -e; chown 999:999 /var/run/mysqld; chmod 0770 /var/run/mysqld; ls -ld /var/run/mysqld"
        ],
        "securityContext": {
          "runAsUser": 0,
          "runAsGroup": 0
        },
        "volumeMounts": [
          {
            "name": "mysql-socket",
            "mountPath": "/var/run/mysqld"
          }
        ]
      }
    }
  ]'

# Fail the installation script if the patched DB still cannot become ready.
kubectl -n mlrun rollout status deployment/mlrun-db --timeout=300s

#####################################################################
# Credentials for pods to pull images
echo "===> Recreating secret, ECR pull credentials for k8s jobs expire every 12 hours"
kubectl --namespace mlrun delete secret ecr-pull-secret
kubectl --namespace mlrun create secret docker-registry ecr-pull-secret \
  --docker-server=$ECR_SERVER \
  --docker-username=AWS \
  --docker-password=$(aws ecr get-login-password --region us-east-1)

# Credentials for literal secret (for AWS python SDK)
kubectl --namespace mlrun delete secret aws-creds-literal
kubectl --namespace mlrun create secret generic aws-creds-literal \
 --from-literal=AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
 --from-literal=AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
 --from-literal=REGION=us-east-1
