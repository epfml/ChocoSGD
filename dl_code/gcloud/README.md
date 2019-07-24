A learning note for the usage of Kubernetes Engine.

---

# Quickstart
## Configuring default settings for gcloud
* Setting a default project: `gcloud config set project [PROJECT_ID]`.
* Setting a default compute zone: `gcloud config set compute/zone [COMPUTE_ZONE]`.

## Creating a Kubernetes Engine cluster
A cluster consists of at least one cluster master machine and multiple worker machines called nodes. Nodes are Compute Engine virtual machine (VM) instances that run the Kubernetes processes necessary to make them part of the cluster. You deploy applications to clusters, and the applications run on the nodes.

To create a cluster, run the following command: `gcloud container clusters create [CLUSTER_NAME]`, where `[CLUSTER_NAME]` is the name you choose for the cluster.

### Get authentication credentials for the cluster
After creating your cluster, you need to get authentication credentials to interact with the cluster.

To authenticate for the cluster, run the following command: `gcloud container clusters get-credentials [CLUSTER_NAME]`

### Deploying an application to the cluster
Now that you have created a cluster, you can deploy a containerized application to it.

#### Creating the Deployment
To run `hello-app` in your cluster, run the following command: `kubectl run hello-server --image gcr.io/google-samples/hello-app:1.0 --port 8080`.
This Kubernetes command, `kubectl run`, creates a new Deployment named `hello-server`. The Deployment's Pod runs the `hello-app` image in its container.

#### Exposing the Deployment
After deploying the application, you need to expose it to the Internet so that users can access it. You can expose your application by creating a Service, a Kubernetes resource that exposes your application to external traffic.

To expose your application, run the following kubectl expose command:
`kubectl expose deployment hello-server --type "LoadBalancer"`.
Passing in the `--type "LoadBalancer"` flag creates a Compute Engine load balancer for your container. Load balancers are billed per Compute Engine's load balancer pricing.

#### Inspecting and viewing the application
1. Inspect the `hello-server` Service by running `kubectl get`: `kubectl get service hello-server`.
From this command's output, copy the Service's external IP address from the `EXTERNAL IP` column.
2. View the application from your web browser using the external IP address with the exposed port: `http://[EXTERNAL_IP]:8080`.

### Clean up
To avoid incurring charges to your Google Cloud Platform account for the resources used in this quickstart:
1. Delete the application's Service by running `kubectl delete service hello-server`.
2. Delete your cluster by running `gcloud container clusters delete [CLUSTER_NAME]`.

### Switch from different contents
You can switch from local to gcloud and back with:
`kubectl config use-context CONTEXT_NAME`
to list all cotexts: `kubectl config get-contexts`.

You can create different enviroments for local and gcloud and put it in separate yaml files.


# How-to guides
## [Creating a cluster](https://cloud.google.com/kubernetes-engine/docs/how-to/creating-a-container-cluster)
### Creating a cluster
To create a container cluster with the `gcloud` command-line tool, use the gcloud container clusters command:
```
gcloud container clusters create [CLUSTER_NAME] [--zone [COMPUTE_ZONE]]
```
where `[CLUSTER_NAME]` is the name you choose for the cluster. The optional `--zone` flag overrides the default `compute/zone` property set by `gcloud config set compute/zone`.

Below are some optional flags that you can specify:
* `--additional-zones`
* `--cluster-version`
* `--enable-basic-auth`
* `--image-type`:
    * The base `node image` that nodes in the cluster runs on. To see the default image type and the list of available image types, run `gcloud container get-server-config`.
    * When you create a Kubernetes Engine cluster or node pool, you can choose the operating system image that runs on each node. You can also upgrade an existing cluster to use a different node image type.
    * Kubernetes Engine offers the following node image options for your cluster:
        * Container-Optimized OS from Google
        * Ubuntu
    * Example: `gcloud container clusters create [CLUSTER_NAME] --image-type ubuntu --cluster-version 1.6.4`
* `--machine-type`:
    * Compute Engine machine type to use for nodes in the cluster. If omitted, the default machine type is `n1-standard-1`.
* `--num-nodes`:
    * The number of nodes to create in the cluster. You must have available resource quota for the nodes and their resources.

### Viewing your clusters
To view a specific cluster, run the following command: `gcloud container clusters describe [CLUSTER_NAME]`.

To view all clusters in a specific zone: `gcloud container clusters list`.

### Setting the default cluster
If you have multiple clusters, you need to set a default cluster for the `gcloud` and `kubectl` command-line tools:
1. set the default cluster for `gcloud` by running the following command: `gcloud config set container/cluster [CLUSTER_NAME]`.
2. pass the cluster's credentials to `kubectl`: `gcloud container clusters get-credentials [CLUSTER_NAME]`.
    * This command adds the cluster's authentication credentials to the kubeconfig file in your environment.
3. To ensure that kubectl has the proper credentials, run `gcloud auth application-default login`.
    * kubectl uses Application Default Credentials to authenticate to the cluster.

### Passing cluster credentials to kubectl
When you create a cluster using `gcloud`, the cluster's authentication credentials are added to the local `kubeconfig` file.
If you created a cluster using GCP Console or using `gcloud` on a different machine, you need to make the cluster's credentials available to `kubectl` in your current environment.

To pass the cluster's credentials to kubectl, run the following command: `gcloud container clusters get-credentials [CLUSTER_NAME]`.

## [Upgrading a Container Cluster](https://cloud.google.com/kubernetes-engine/docs/how-to/upgrading-a-container-cluster)

## [Resizing a Container Cluster](https://cloud.google.com/kubernetes-engine/docs/how-to/resizing-a-container-cluster)
### Resizing a container cluster
To resize your cluster's node pools, you use the gcloud container clusters resize command.

You must specify the cluster's name, the name of the desired node pool, and the new number of nodes:
`gcloud container clusters resize [CLUSTER_NAME] --node-pool [NODE_POOL] --size [SIZE]`. Repeat this command for each node pool. If your cluster only has its default node pool, omit the --node-pool flag.

### Increasing the size of your cluster
When you increase the size of a cluster:
* New node instances are created using the same configuration as the existing instances
* New Pods may be scheduled onto the new instances
* Existing Pods are not moved onto the new instances

When you increase the size of a node pool that spans multiple zones, the new size represents the number of nodes in the node pool per zone. For example, if you have a node pool of size 2 spanning two zones, the total node count is 4. If you resize the node pool to size 4, the total node count becomes 8.

### Decreasing the size of your cluster
When you decrease the size of a cluster:
* The Pods that are scheduled on the instances being removed are killed
* Pods managed by a replication controller are rescheduled by the controller onto the remaining instances
* Pods not managed by a replication controller are not restarted

## [Setting up Logging with Stackdriver](https://cloud.google.com/kubernetes-engine/docs/how-to/logging)
You can enable [Stackdriver Logging](https://cloud.google.com/logging/docs) to have Kubernetes Engine automatically collect, process, and store your container and system logs in a dedicated, persistent datastore.
* **Container logs** are collected from your containers.
* **System logs** are collected from the cluster's components, such as docker and kubelet.
* **Events** are logs about activity in the cluster, such as the scheduling of Pods.

While Kubernetes Engine itself stores logs, these logs are not stored permanently.

For container and system logs, Kubernetes Engine deploys a per-node logging agent that reads container logs, adds helpful metadata, and then stores them.

### Enabling Stackdriver Logging
You can create a cluster with Stackdriver Logging enabled, or enable Stackdriver Logging in an existing cluster.

#### Creating a cluster with logging
When you create a cluster, the `--enable-cloud-logging` flag is automatically set, which enables Stackdriver Logging in the cluster.

To disable this default behavior, set the `--no-enable-cloud-logging` flag.

#### Enabling logging for an existing cluster
To enable logging for an existing cluster, run the following command, where [CLUSTER_NAME] is the name of the cluster.
```
gcloud beta container clusters update [CLUSTER_NAME] --logging-service logging.googleapis.com
```

## [Setting up Monitoring with Stackdriver](https://cloud.google.com/kubernetes-engine/docs/how-to/monitoring)
Stackdriver monitors system metrics and custom metrics.
* **System metrics** are measurements of the cluster's infrastructure, such as CPU or memory usage.
* **Custom metrics** are application-specific metrics that you define yourself, such as the total number of active user sessions or the total number of rendered pages.

> For system metrics, Stackdriver creates a Deployment that periodically connects to each node and collects metrics about its Pods and containers, then sends the metrics to Stackdriver. For a list of the system metrics collected from Kubernetes Engine, refer to Metrics List in the Stackdriver documentation.

### Enabling Stackdriver Monitoring
You can create a new cluster with monitoring enabled, or add monitoring capability to an existing cluster.

> Note that your cluster's node pools (including the default node pool) must have the necessary GCP scope to interact with Stackdriver Monitoring (the [scope](https://www.googleapis.com/auth/monitoring)). When you create a new cluster with monitoring, Kubernetes Engine sets this scope automatically; however, existing clusters might not have the necessary permissions.

#### Creating a cluster with monitoring
When you create a cluster, the `--enable-cloud-monitoring` flag is automatically set, which enables Stackdriver Monitoring in the cluster.

To disable this default behavior, set the `--no-enable-cloud-monitoring` flag.

#### Enabling monitoring for an existing cluster
To enable monitoring for an existing cluster, run the following command, where [CLUSTER_NAME] is the name of the cluster.

### Viewing metrics
You can view metrics in the in the GCP Console's [Stackdriver Monitoring menu](https://console.cloud.google.com/monitoring?_ga=2.219089625.-1564962953.1511824460).

## Configuring Cluster Networking
### [Internal Load Balancing](https://cloud.google.com/kubernetes-engine/docs/how-to/internal-load-balancing)
Internal load balancing makes your cluster's services accessible to applications running on the same network but outside of the cluster.

> For example, if you run a cluster alongside some Compute Engine VM instances in the same network and you would like your cluster-internal services to be available to the cluster-external instances, you need to configure one of your cluster's Service resources to add an internal load balancer.

Without internal load balancing, you would need to set up external load balancers, create firewall rules to limit the access, and set up network routes to make the IP address of the application accessible outside of the cluster.

### [Setting up IP Aliasing](https://cloud.google.com/kubernetes-engine/docs/how-to/ip-aliases)
With IP aliases, Kubernetes Engine clusters can allocate Pod IP addresses from a CIDR block known to Google Cloud Platform. This allows your cluster to interact with other Cloud Platform products and entities, and also allows more scalable clusters.
* Pod IPs are reserved within the network ahead of time, which prevents conflict with other compute resources.
* The networking layer can perform anti-spoofing checks to ensure that egress traffic is not sent with arbitrary source IPs.
* Pod IPs are natively routable within the Google Cloud Platform network (including via VPC Network Peering), so clusters can scale to larger sizes faster without using up route quota.
* Aliased IPs can be announced through BGP by the cloud-router, enabling better support for connecting to on-premises networks.
* Firewall controls for Pods can be applied separately from their hosting node.
* IP aliases allow Pods to directly access hosted services without using a NAT gateway.

### [Using an IP Masquerade Agent](https://cloud.google.com/kubernetes-engine/docs/how-to/ip-masquerade-agent)
IP masquerading is a form of network address translation (NAT) used to perform many-to-one IP address translations. Masquerading masks multiple source IP addresses behind a single address.

Using IP masquerading in your clusters increases their security by preventing individual Pod IP addresses from being exposed to traffic outside link-local and additional arbitrary IP ranges. Additionally, it allows you configure communication between IP ranges without masquerade, such as a Pod in the `192.168.0.0/16` range interacting with networking resources in the `10.0.0.0/8` range.

### [Setting a Cluster Network Policy](https://cloud.google.com/kubernetes-engine/docs/how-to/network-policy)
You can use Kubernetes Engine's **network policy enforcement** to control the communication between your cluster's Pods and Services. To define a network policy on Kubernetes Engine, you can use the Kubernetes Network Policy API to create Pod-level firewall rules. These firewall rules determine which Pods and Services can access one another inside your cluster.

Defining network policy helps you enable things like defense in depth when your cluster is serving a multi-level application.
> For example, you can create a network policy to ensure that a compromised front-end service in your application cannot communicate directly with a billing or accounting service several levels down.

Network policy can also make it easier for your application to host data from multiple users simultaneously.
> For example, you can provide secure multi-tenancy by defining a tenant-per-namespace model. In such a model, network policy rules can ensure that Pods and Services in a given namespace cannot access other Pods or Services in a different namespace.

# [GPUs on Kubernetes Engine](https://cloud.google.com/kubernetes-engine/docs/concepts/gpus)
## Limitations
Support for NVIDIA GPUs on Kubernetes Engine has the following limitations:
* GPUs are only supported for the Container-Optimized OS node image.
* You cannot add GPUs to existing node pools.
* GPU nodes cannot be live migrated during maintenance events.
* GPU nodes run the NVIDIA GPU device plugin system addon and have the DevicePlugins Kubernetes alpha feature enabled. Kubernetes Engine automatically manages this device plugin, but Google does not provide support for any third-party device plugins.

### Availability
GPUs are available in specific regions. For a complete list of applicable regions and zones, refer to GPUs on Compute Engine. To see a list of accelerator types supported in each zone, run the following command:
```
gcloud beta compute accelerator-types list
```

## Creating a cluster with GPUs
You create a cluster that runs GPUs in its default node pool using GCP Console or the gcloud command-line tool.
```
gcloud beta container clusters create [CLUSTER_NAME] \
    --accelerator type=[GPU_TYPE],count=[AMOUNT] \
    --zone [COMPUTE_ZONE] --cluster-version [CLUSTER_VERSION]
```
where
* `[CLUSTER_NAME]` is the name you choose for the cluster.
* `[GPU_TYPE]` is the GPU type, either `nvidia-tesla-p100` or `nvidia-tesla-k80`.
* `[AMOUNT]` is the number GPUs to attach to every node in the default node pool.
* `[COMPUTE_ZONE]` is the cluster's compute zone, such as `us-central1-a`.
* `[CLUSTER_VERSION]` is Kubernetes Engine version 1.9.0 or later.

### Installing NVIDIA GPU device drivers
After adding GPU nodes to your cluster, you need to install NVIDIA's device drivers to the nodes. Google provides a DaemonSet that automatically installs the drivers for you.

To deploy the installation DaemonSet, run the following command:
```
kubectl create -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/k8s-1.9/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```
The installation takes several minutes to complete. Once installed, the NVIDIA GPU device plugin surfaces NVIDIA GPU capacity via Kubernetes APIs.

### Configuring Pods to consume GPUs
Below is an example of a Pod specification that consumes GPUs:
```
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: my-gpu-container
    resources:
      limits:
       nvidia.com/gpu: 2
```
If you want to use multiple GPU accelerator types per cluster, you must create multiple node pools, each with their own accelerator type. Kubernetes Engine attaches a unique node selector to GPU nodes to help place GPU workloads on nodes with specific GPU types:
* **Key**: `cloud.google.com/gke-accelerator`
* **Value**: `nvidia-tesla-k80` or `nvidia-tesla-p100`

You can target particular GPU types by adding this node selector to your workload's Pod specification. For example:
```
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: my-gpu-container
    resources:
      limits:
       nvidia.com/gpu: 2
  nodeSelector:
    cloud.google.com/gke-accelerator: nvidia-tesla-p100 # or nvidia-tesla-k80
```

### Creating an autoscaling GPU node pool
To take the best, most cost-effective advantage of GPUs on Kubernetes Engine, and to take advantage of cluster autoscaling, we recommend creating separate GPU node pools in your clusters.

When you add a GPU node pool to an existing cluster that already runs a non-GPU node pool, Kubernetes Engine automatically taints the GPU nodes with the following node taint:
* **Key**: `nvidia.com/gpu`
* **Effect**: `NoSchedule`

Additionally, Kubernetes Engine automatically applies the corresponding tolerations to Pods requesting GPUs by running the ExtendedResourceToleration admission controller.

This causes only Pods requesting GPUs to be scheduled on GPU nodes, which enables more efficient autoscaling: your GPU nodes can quickly scale down if there are not enough Pods requesting GPUs.

You create a GPU node pool in an existing cluster using GCP Console or the `gcloud` command-line tool.
```
gcloud beta container node-pools create [POOL_NAME] \
    --accelerator type=[GPU_TYPE],count=[AMOUNT] --zone [COMPUTE-ZONE] \
    --cluster [CLUSTER-NAME] [--num-nodes 3 --min-nodes 2 --max-nodes 5 \
    --enable-autoscaling]
```
where
* `[POOL_NAME]` is the name you choose for the node pool.
* `[GPU_TYPE]` is the GPU type, either `nvidia-tesla-p100` or `nvidia-tesla-k80`.
* `[AMOUNT]` is the number GPUs to attach to nodes in the node pool.
* `[COMPUTE_ZONE]` is the cluster's compute zone, such as `us-central1-a`.
* `[CLUSTER-NAME]` is the name of the cluster in which to create the node pool.
* `--num-nodes` specifies the initial number of nodes to be created.
* `--min-nodes` specifies the minimum number of nodes to run any given time.
* `--max-nodes` specifies the maximum number of nodes that can run.
* `--enable-autoscaling` allows the node pool to autoscale when workload demand changes.

### [Monitoring GPU nodes](https://cloud.google.com/monitoring/docs/)
Kubernetes Engine exposes the following Stackdriver Monitoring metrics for containers using GPUs. You can use these metrics to monitor how your GPU workloads perform:
* `container/accelerator/duty_cycle`
* `container/accelerator/memory_total`
* `container/accelerator/memory_used`

# [Google Cloud Storage](https://cloud.google.com/storage)
## [Cloud Storage FUSE](https://cloud.google.com/storage/docs/gcs-fuse)
Cloud Storage FUSE is an open source FUSE adapter that allows you to mount Cloud Storage buckets as file systems on Linux or OS X systems. It also provides a way for applications to upload and download Cloud Storage objects using standard file system semantics. Cloud Storage FUSE can be run anywhere with connectivity to Cloud Storage, including Google Compute Engine VMs or on-premises systems1.
