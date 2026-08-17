import argparse
import json
import logging
import os
import sys

from wqb_agent import Agent, WQBClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("wqb.main")


def load_config(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="WQB Alpha self-evolving research agent"
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to config JSON (default: config.json)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Number of research rounds to run (overrides config)",
    )
    parser.add_argument(
        "--state-dir",
        default=None,
        help="Directory for memory/trajectory state (overrides config)",
    )
    parser.add_argument(
        "--single-round",
        action="store_true",
        help="Run exactly one research round and exit",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if config is None:
        example = os.path.join(os.path.dirname(__file__), "config.example.json")
        print(
            f"Config file '{args.config}' not found. "
            f"Copy {example} to {args.config} and edit it."
        )
        sys.exit(1)

    if args.state_dir:
        config["agent"]["state_dir"] = args.state_dir

    try:
        client = WQBClient()
    except Exception as exc:
        print(f"Credentials error: {exc}")
        sys.exit(1)

    agent = Agent(client, config)
    if args.single_round:
        agent.run_one_round(1)
    else:
        rounds = args.rounds or config["agent"].get("max_rounds", 5)
        agent.run(max_rounds=rounds)


if __name__ == "__main__":
    main()
