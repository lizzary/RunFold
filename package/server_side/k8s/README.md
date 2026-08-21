# Kubernetes deployment

The server must run with one replica and one worker because SQLite and the embedded
LanceDB index share one local data directory. The `Recreate` strategy prevents an
old and a new pod from serving concurrently during an update.

1. Copy `config.example.yaml` to a file outside version control, replace every
   placeholder, and verify that the embedding model and dimensions match.
2. Build and push the image, then replace the image in `runfold.yaml` with its
   immutable registry tag or digest.
3. Create the configuration Secret and apply the workload:

```sh
docker build -t REGISTRY/runfold-server:TAG .
docker push REGISTRY/runfold-server:TAG
kubectl create secret generic runfold-config \
  --from-file=config.yaml=/secure/path/config.yaml
kubectl apply -f k8s/runfold.yaml
kubectl rollout status deployment/runfold-server
kubectl get pods,service,pvc
```

The PVC storage class must provide filesystem semantics suitable for SQLite and
LanceDB. Keep `replicas: 1`; do not share the PVC with another server process.
After the first successful startup, remove `auth.bootstrap_admin` from the source
configuration and recreate the Secret. Expose the ClusterIP Service through an
HTTPS ingress appropriate for the cluster.
