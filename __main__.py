import argparse
import server
import json
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="The config file to use")
    flags, _ = parser.parse_known_args()
    if not flags.config:
        flags.config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
    with open(flags.config) as f:
        config = json.loads(f.read())
    server.start(config)

if __name__ == "__main__":
    main()