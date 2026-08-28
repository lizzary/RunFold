# Kubernetes deployment

The server must run with one replica and one worker because SQLite and the embedded
LanceDB index share one local data directory. The `Recreate` strategy prevents an
old and a new pod from serving concurrently during an update.

## Docker Desktop Kubernetes

The checked-in manifest uses the local image `runfold-server:root` with
`imagePullPolicy: Never`. Build the image, import it into the Docker Desktop
Kubernetes node's containerd store, create the configuration Secret from the
local `config.yaml`, and apply the workload from `package/server_side`:

```powershell
kubectl config use-context docker-desktop
docker build --tag runfold-server:root .

cmd /c "docker image save runfold-server:root | docker exec -i desktop-control-plane ctr --namespace k8s.io images import -"

kubectl create secret generic runfold-config `
  --from-file=config.yaml=config.yaml `
  --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -f k8s/runfold.yaml
kubectl rollout restart deployment/runfold-server
kubectl rollout status deployment/runfold-server --timeout=120s
kubectl get deployment,pods,service,pvc
```

Docker Desktop Kubernetes uses an image store separate from the normal Docker
image store. Repeat the image import and Secret creation after resetting or
recreating the Kubernetes cluster. Re-import the image after every rebuild, then
restart the Deployment.

If rollout does not complete within the timeout, inspect the pod instead of
waiting without a deadline:

```powershell
kubectl get pods -l app.kubernetes.io/name=runfold-server -o wide
kubectl describe pod -l app.kubernetes.io/name=runfold-server
kubectl get events --sort-by=.lastTimestamp
kubectl logs deployment/runfold-server --tail=100
```

For local access from Windows:

```powershell
kubectl port-forward service/runfold-server 8383:8383
```

Then request `http://127.0.0.1:8383/health/ready`.

## Registry deployment

For a remote cluster, copy `config.example.yaml` to a file outside version
control, replace every placeholder, verify that the embedding model and
dimensions match, and change `runfold.yaml` to a pushed immutable image tag or
digest with `imagePullPolicy: IfNotPresent`.

```sh
docker build -t REGISTRY/runfold-server:TAG .
docker push REGISTRY/runfold-server:TAG
kubectl create secret generic runfold-config \
  --from-file=config.yaml=/secure/path/config.yaml \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f k8s/runfold.yaml
kubectl rollout status deployment/runfold-server --timeout=120s
kubectl get pods,service,pvc
```

The PVC storage class must provide filesystem semantics suitable for SQLite and
LanceDB. Keep `replicas: 1`; do not share the PVC with another server process.
After the first successful startup, remove `auth.bootstrap_admin` from the source
configuration and recreate the Secret. Expose the ClusterIP Service through an
HTTPS ingress appropriate for the cluster.
