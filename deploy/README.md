# EpochDB Multi-Cloud Deployment Guide

This guide describes how to deploy the **EpochDB Distributed Server** to production environments on **AWS (EKS)**, **GCP (GKE)**, and **Azure (AKS)**. 

---

## Production Cluster Architecture

The production setup uses a Kubernetes-native model to manage EpochDB's sharded architecture:
- **`epochdb-shard` (StatefulSet)**: Multi-replica storage group. Each shard gets a stable, dedicated pod DNS record (e.g. `epochdb-shard-0.epochdb-shard-service`) via a Kubernetes Headless Service, and a dedicated Persistent Volume (AWS EBS, GCP Persistent Disk, or Azure Disk).
- **`epochdb-coordinator` (Deployment)**: Stateless gateway group that acts as the entry point, proxies/routes requests, and implements consistent hashing. Can scale dynamically.
- **Service Exposer**: Exposes the Coordinator externally via a standard cloud `LoadBalancer` Service.

---

## Step 1: Provision Infrastructure (Terraform)

Choose your cloud provider and use the provided Terraform configurations to set up the VPC, Kubernetes cluster, and high-performance SSD storage class.

### AWS (EKS)
1. Navigate to: `deploy/terraform/aws`
2. Run commands:
   ```bash
   terraform init
   terraform apply
   ```
3. Update local kubeconfig to point to your new EKS cluster:
   ```bash
   aws eks update-kubeconfig --name epochdb-cluster --region us-east-1
   ```

### GCP (GKE)
1. Navigate to: `deploy/terraform/gcp`
2. Run commands:
   ```bash
   terraform init
   terraform apply -var="gcp_project_id=your-project-id"
   ```
3. Configure local credentials:
   ```bash
   gcloud container clusters get-credentials epochdb-cluster --region us-central1
   ```

### Azure (AKS)
1. Navigate to: `deploy/terraform/azure`
2. Run commands:
   ```bash
   terraform init
   terraform apply
   ```
3. Import the cluster configuration:
   ```bash
   az aks get-credentials --resource-group epochdb-resources --name epochdb-cluster
   ```

---

## Step 2: Build and Push Docker Image

Before deploying the manifests, build the server image and push it to your private container registry (ECR, Artifact Registry, or ACR).

```bash
# 1. Build the Docker Image
docker build -t epochdb-server:latest -f Dockerfile .

# 2. Tag for your Registry (Example: AWS ECR)
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
docker tag epochdb-server:latest ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/epochdb:latest

# 3. Log in and Push
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/epochdb:latest
```

---

## Step 3: Configure Credentials & Storage

1. **Secrets (`deploy/kubernetes/secrets.yaml`)**:
   Open the file and encode your secure tokens using Base64.
   ```bash
   echo -n "your-secure-client-api-key" | base64
   echo -n "your-secure-internal-shard-token" | base64
   ```
   Replace `API_KEY` and `INTERNAL_AUTH_TOKEN` in the manifest.

2. **Storage Classes (`deploy/kubernetes/shards.yaml`)**:
   Uncomment or configure the `storageClassName` matching your provider's SSD storage class created by Terraform:
   * **AWS**: Set `storageClassName: gp3`
   * **GCP**: Set `storageClassName: pd-ssd`
   * **Azure**: Set `storageClassName: managed-premium-ssd`

3. **Image Path**:
   In `shards.yaml` and `coordinator.yaml`, replace `image: jersobh/epochdb:latest` with your newly pushed registry image path.

---

## Step 4: Deploy to Kubernetes

Deploy the manifests in order:

```bash
# 1. Create Namespace
kubectl apply -f deploy/kubernetes/namespace.yaml

# 2. Apply ConfigMap and Secret Credentials
kubectl apply -f deploy/kubernetes/secrets.yaml

# 3. Spin up the Stateful Shards
kubectl apply -f deploy/kubernetes/shards.yaml

# 4. Spin up the Coordinator Gateway
kubectl apply -f deploy/kubernetes/coordinator.yaml
```

---

## Step 5: Verification

1. **Verify Resources**:
   Ensure all replicas are `Running` and healthy:
   ```bash
   kubectl get all -n epochdb
   ```

2. **Retrieve the Gateway IP**:
   Get the LoadBalancer IP/Hostname assigned to the coordinator:
   ```bash
   kubectl get service epochdb-coordinator-service -n epochdb
   ```

3. **Perform a Health Check**:
   ```bash
   # Replace with the external IP/hostname and your API key
   curl -H "X-API-Key: test-api-key-12345" http://<LOAD_BALANCER_EXTERNAL_IP>:8080/healthz
   ```

4. **Integration Test (Python SDK)**:
   Ensure clients can connect using the public URL:
   ```python
   import asyncio
   from epochdb import AsyncRemoteEpochDB

   async def test_conn():
       # Connect to the cloud LoadBalancer
       db = AsyncRemoteEpochDB(host="<LOAD_BALANCER_EXTERNAL_IP>", port=8080, api_key="test-api-key-12345")
       
       # Write check
       mem_id = await db.remember("Test connection to Kubernetes cluster.")
       print(f"Memory stored successfully at ID: {mem_id}")
       
       await db.close()

   asyncio.run(test_conn())
   ```
