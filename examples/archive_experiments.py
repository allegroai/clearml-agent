#!/usr/bin/python3
"""
An example script that cleans up failed experiments by moving them to the archive
"""
import argparse
from datetime import datetime

from trains_agent import APIClient

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--project", "-P", help="Project ID. Only clean up experiments from this project")
parser.add_argument("--user", "-U", help="User ID. Only clean up experiments assigned to this user")
parser.add_argument(
    "--status", "-S", default="failed",
    help="Experiment status. Only clean up experiments with this status (default %(default)s)"
)
parser.add_argument(
    "--iterations", "-I", type=int,
    help="Number of iterations. Only clean up experiments with less or equal number of iterations"
)
parser.add_argument(
    "--sec-from-start", "-T", type=int,
    help="Seconds from start time. "
         "Only clean up experiments if less or equal number of seconds have elapsed since started"
)

args = parser.parse_args()

client = APIClient()

tasks = client.tasks.get_all(
    project=[args.project] if args.project else None,
    user=[args.user] if args.user else None,
    status=[args.status] if args.status else None,
    system_tags=["-archived"]
)

count = 0

for task in tasks:
    if args.iterations and (task.last_iteration or 0) > args.iterations:
        continue
    if args.sec_from_start:
        if not task.started:
            continue
        if (datetime.utcnow() - task.started.replace(tzinfo=None)).total_seconds() > args.sec_from_start:
            continue

    try:
        client.tasks.edit(
            task=task.id,
            system_tags=(task.system_tags or []) + ["archived"],
            force=True
        )
        count += 1
    except Exception as ex:
        print("Failed editing experiment: {}".format(ex))

print("Cleaned up {} experiments".format(count))
