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
kubectl --namespace mlrun delete secret ecr-build-secret
kubectl --namespace mlrun create secret generic ecr-build-secret \
  --from-file=./k8s/credentials
 # Literal secret does not work
#  --from-literal=aws_access_key_id=AKIA... \

echo "===> Installing mlrun with helm..."
# helm --namespace mlrun \
#     install mlrun-ce \
#     --version 0.11.0 \
#     --wait \
#     --timeout 3600s \
#     --set global.registry.url=$ECR_SERVER \
#     --set global.registry.secretName=ecr-build-secret \
#     --set global.externalHostAddress=localhost \
#     --set pipelines.enabled=false \
#     --set kube-prometheus-stack.enabled=false \
#     --set spark-operator.enabled=false \
#     --set mlrun-db.initContainers[0].name="fix-permissions" \
#     --set mlrun-db.initContainers[0].image="alpine" \
#     --set mlrun-db.initContainers[0].command[0]="sh" \
#     --set mlrun-db.initContainers[0].command[1]="-c" \
#     --set mlrun-db.initContainers[0].command[2]="mkdir -p /var/run/mysqld && chown -R 999:999 /var/run/mysqld /var/lib/mysql" \
#     --set mlrun-db.initContainers[0].volumeMounts[0].name="data" \
#     --set mlrun-db.initContainers[0].volumeMounts[0].mountPath="/var/lib/mysql" \
#     --set mlrun-db.initContainers[0].volumeMounts[1].name="run" \
#     --set mlrun-db.initContainers[0].volumeMounts[1].mountPath="/var/run/mysqld" \
#     mlrun-ce/mlrun-ce

# 1. Fix the registry syntax and run helm in the background (&)
helm --namespace mlrun \
    install mlrun-ce \
    --version 0.11.0 \
    --set global.registry=$ECR_SERVER \
    --set pipelines.enabled=false \
    --set kube-prometheus-stack.enabled=false \
    --set spark-operator.enabled=false \
    mlrun-ce/mlrun-ce &
    
HELM_PID=$!

# 2. Poll the cluster until Helm creates the mlrun-db deployment
echo "Waiting for mlrun-db deployment to be created..."
while ! kubectl get deployment mlrun-db -n mlrun > /dev/null 2>&1; do
  sleep 2
done

# 3. Force the database to use MySQL 8.0 to bypass the 8.4 breaking changes
kubectl set image deployment/mlrun-db mlrun-db=mysql:8.0 -n mlrun

# 4. Wait for the background Helm process and its hooks to finish successfully
wait $HELM_PID

# 5. Verify the rollouts
kubectl rollout status deployment/mlrun-db -n mlrun --timeout=600s
kubectl rollout status deployment/mlrun-api-chief -n mlrun --timeout=600s

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
