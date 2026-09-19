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

echo "===> Installing mlrun with helm...inject writable volume and wait"
# 1. Start Helm in the background so it does not block the terminal
helm --namespace mlrun \
    install mlrun-ce \
    --version 0.11.0 \
    --timeout 1200s \
    --set global.registry.url=$ECR_SERVER \
    --set global.registry.secretName=ecr-build-secret \
    --set global.externalHostAddress=$(minikube ip) \
    --set pipelines.enabled=false \
    --set kube-prometheus-stack.enabled=false \
    --set spark-operator.enabled=false \
    mlrun-ce/mlrun-ce &

HELM_PID=$!

# 2. Wait up to 2 minutes for Helm to create the mlrun-db deployment
echo "Waiting for mlrun-db deployment to be created..."
for i in {1..60}; do
  if kubectl get deployment mlrun-db -n mlrun > /dev/null 2>&1; then
    break
  fi
  sleep 2
done

# 3. Extract the exact container name dynamically
CONTAINER_NAME=$(kubectl get deployment mlrun-db -n mlrun -o jsonpath='{.spec.template.spec.containers[0].name}')

# 4. Inject the emptyDir volume using a Strategic Merge Patch
echo "Patching $CONTAINER_NAME to fix socket permissions..."
kubectl patch deployment mlrun-db -n mlrun --patch "
spec:
  template:
    spec:
      volumes:
      - name: mysql-socket
        emptyDir: {}
      containers:
      - name: ${CONTAINER_NAME}
        volumeMounts:
        - name: mysql-socket
          mountPath: /var/run/mysqld
"

# 5. Bring Helm back to the foreground and wait for it to complete
echo "Volume injected! Waiting for Helm installation to finish..."
wait $HELM_PID

# 6. Explicitly wait for the deployments to be fully ready
echo "Installation complete, verifying readiness..."
kubectl wait --namespace mlrun --for=condition=Available deployment/mlrun-db --timeout=600s
kubectl wait --namespace mlrun --for=condition=Available deployment/mlrun-api-chief --timeout=10s

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
