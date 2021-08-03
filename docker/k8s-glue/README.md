This folder contains an example docker and templates for running the k8s glue as a pod in a k8s cluster

Please note that ClearML credentials and server addresses should either be filled in the clearml.conf file before
 building the glue docker or provided in the k8s-glue.yml template.

To run, you'll need to:
* Create a secret from pod_template.yml:
  ```bash
  kubectl -n clearml create secret generic k8s-glue-pod-template --from-file=pod_template.yml
  ```
* Apply the k8s glue template:
  ```bash
  kubectl -n clearml apply -f k8s-glue.yml
  ```
  