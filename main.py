import os.path

import yaml

from experiment import multiple_runs, run
from utils import get_parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file {args.config} not found")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # save cli config to a yaml file
    with open(f"{args.run_dir}/cli_config.yaml", "w") as f:
        yaml.dump(vars(args), f)

    if args.n_runs > 1:
        multiple_runs(config, args)
    else:
        run(1, config, args, True)


if __name__ == "__main__":
    main()
