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
    return parser.parse_args()


def main():
    args = parse_args()
    k8s = K8sIntegration(ports_mode=args.ports_mode, num_of_services=args.num_of_services)
    k8s.k8s_daemon(args.queue)


if __name__ == "__main__":
    main()
