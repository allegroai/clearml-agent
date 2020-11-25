"""
This example assumes you have preconfigured services with selectors in the form of
 "ai.allegro.agent.serial=pod-<number>" and a targetPort of 10022.
The K8sIntegration component will label each pod accordingly.
"""
from argparse import ArgumentParser

from trains_agent.glue.k8s import K8sIntegration


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--queue", type=str, help="Queue to pull tasks from"
    )
    parser.add_argument(
        "--ports-mode", action='store_true', default=False,
        help="Ports-mode will add a label to the pod which can be used in services in order to expose ports"
    )
    parser.add_argument(
        "--num-of-services", type=int, default=20,
        help="Specify the number of k8s services to be used. Use only with ports-mode."
    )
    parser.add_argument(
        "--base-port", type=int,
        help="If using ports-mode, specifies the base port exposed by the services."
             "For pod #X, the port will be <base-port>+X"
    )
    parser.add_argument(
        "--pod-trains-conf", type=str,
        help="Configuration file to be used by the pod itself (if not provided, current configuration is used)"
    )
    parser.add_argument(
        "--overrides-yaml", type=str,
        help="YAML file containing pod overrides to be used when launching a new pod"
    )
    parser.add_argument(
        "--template-yaml", type=str,
        help="YAML file containing pod template. If provided pod will be scheduled with kubectl apply "
             "and overrides are ignored, otherwise it will be scheduled with kubectl run"
    )
    parser.add_argument(
        "--ssh-server-port", type=int, default=0,
        help="If non-zero, every pod will also start an SSH server on the selected port (default: zero, not active)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    user_props_cb = None
    if args.ports_mode and args.base_port:
        def user_props_cb(pod_number):
            return {"k8s-pod-port": args.base_port + pod_number}

    k8s = K8sIntegration(
        ports_mode=args.ports_mode, num_of_services=args.num_of_services, user_props_cb=user_props_cb,
        overrides_yaml=args.overrides_yaml, trains_conf_file=args.pod_trains_conf, template_yaml=args.template_yaml,
        extra_bash_init_script=K8sIntegration.get_ssh_server_bash(
            ssh_port_number=args.ssh_server_port) if args.ssh_server_port else None
    )
    k8s.k8s_daemon(args.queue)


if __name__ == "__main__":
    main()
