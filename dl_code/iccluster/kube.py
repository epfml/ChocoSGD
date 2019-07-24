"""
Convenience functions for dealing with Kubernetes at MLO

- Listing user's pods or jobs
- Inspecting
- Deleting
- Cleanup of finished items
"""
import re
import subprocess
import os
from pprint import pprint
from typing import Any, Dict, Generator, Union

from kubernetes import client, config
from kubernetes.client import V1Job, V1Pod, V1Status

config.load_kube_config()

USER = os.getenv("USER")
NAMESPACE = "mlo"


def pods(running=None, all_users=False) -> Generator[V1Pod, None, None]:
    """
    Generator for pods
    :param running: if boolean (not None, we will filter by running yes/no)
    """
    v1 = client.CoreV1Api()

    if USER is not None and not all_users:
        label_selector = f"user={USER}"
    else:
        label_selector = ""

    for pod in v1.list_namespaced_pod(NAMESPACE, label_selector=label_selector).items:
        if (
            running is None  # Don't care-mode
            or (running and status(pod) == "running")
            or (not running and status(pod) in ["succeeded", "failed"])
        ):
            yield pod


def jobs(running=None) -> Generator[V1Job, None, None]:
    """
    Generator for jobs
    :param running: if boolean (not None, we will filter by running yes/no)
    """
    v1 = client.BatchV1Api()

    if USER is not None:
        label_selector = f"user={USER}"
    else:
        label_selector = ""

    for job in v1.list_namespaced_job(NAMESPACE, label_selector=label_selector).items:
        if (
            running is None  # Don't care-mode
            or (running and status(job) == "running")
            or (not running and status(job) in ["finished", "failed"])
        ):
            yield job


def status(entry: Union[V1Pod, V1Job]) -> str:
    """
    Figure out what the status is of a Pod or Job
    """
    if isinstance(entry, V1Job):
        job = entry
        completions = job.spec.completions if job.spec.completions else 0
        succeeded = job.status.succeeded if job.status.succeeded else 0
        failed = job.status.failed if job.status.failed else 0
        if succeeded + failed < completions:
            return "running"
        elif succeeded == completions:
            return "completed"
        else:
            return "failed"
    elif isinstance(entry, V1Pod):
        return entry.status.phase.lower()
    else:
        raise ValueError("Unknown object type")


def gpus(pod: V1Pod) -> int:
    count = 0
    for container in pod.spec.containers:
        limits = container.resources.limits
        if limits is not None:
            count += int(limits.get("nvidia.com/gpu", 0))
    return count


def describe(entry: Union[V1Pod, V1Job]) -> Dict[str, Any]:
    """
    Describe an object from this file (either V1Pod or V1Job)
    """
    if isinstance(entry, V1Pod):
        info = {
            "kind": "Pod",
            "metadata.name": entry.metadata.name,
            "metadata.labels": entry.metadata.labels,
            "metadata.creation_timestamp": entry.metadata.creation_timestamp,
            "[status]": status(entry),
            "[gpus]": gpus(entry),
            "status.start_time": entry.status.start_time,
        }
        pprint(info)
        return info
    elif isinstance(entry, V1Job):
        info = {
            "kind": "Job",
            "metadata.name": entry.metadata.name,
            "metadata.labels": entry.metadata.labels,
            "metadata.creation_timestamp": entry.metadata.creation_timestamp,
            "[status]": status(entry),
            "status.start_time": entry.status.start_time,
        }
        pprint(info)
        return info
    else:
        raise ValueError("Unknown object")


def delete(entry: Union[V1Pod, V1Job]) -> V1Status:
    """
    Delete/kill an object from this file (either V1Pod or V1Job)
    """
    if isinstance(entry, V1Pod):
        v1 = client.CoreV1Api()
        return v1.delete_namespaced_pod(
            entry.metadata.name,
            namespace=NAMESPACE,
            body=client.V1DeleteOptions(propagation_policy="Foreground"),
        )
    if isinstance(entry, V1Job):
        v1 = client.BatchV1Api()
        return v1.delete_namespaced_job(
            entry.metadata.name,
            namespace=NAMESPACE,
            body=client.V1DeleteOptions(propagation_policy="Foreground"),
        )
    else:
        raise ValueError("Unknown object")


def cleanup():
    """
    Delete non-running jobs and pods
    """
    for job in jobs(running=False):
        delete(job)
    for pod in pods(running=False):
        delete(pod)


def get_status(all_users=False):
    for pod in pods(running=True, all_users=all_users):
        if gpus(pod) > 0:
            output = subprocess.check_output(
                ["kubectl", "exec", pod.metadata.name, "nvidia-smi"]
            ).decode("utf-8")
            usage = re.findall(r"""\d+\%""", output)
            print(f"{pod.metadata.name:30s}", usage)
