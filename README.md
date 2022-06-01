# emotion-classification-using-transformers
We, robot

### Conda environment stuff
#### installing the environment: 
    `conda env create --file environemtn.yaml`

#### activating: 
    - windows: `activate <env_name>`
    - linux: `source activate <env_name>`

#### adding new packages:
1. Check out the package documentation to out the correct way to install the package
    - preferably, package exists within some conda channel
        - that means it can be installed with `conda install -c <channel_name> <package_name>`
        - specify channel name only if needed, check documentation for more info
    - if not in conda, then pip
      - `pip install <package_name>` 
      - if first time adding pip package within the environment, run `conda install pip` first
2. add the package information to `environment.yml`:
    - if installed using conda: just add into the list of dependencies
    - if installed using pip: add to list of pip dependencies
    - *specify the package version*, it insures that our code works forever and is not broken by package updates
3. run `conda env update --prune` to update the environment and sync with the new state of `environment.yml`
 - conda cheat sheet: https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf

### Running the models
#### Baselines
    - Instructions coming soon
#### Command-line arguments
   - `--config` -- path to the config file (e.g. `config/adapter-test.yaml`)
   - `--device` -- device to use for training (e.g. `cuda`)
   - `--model_name` -- name of the transformer model that will be downloaded from HuggingFace (e.g. `roberta-base`)
   - `--run_name` -- name of the experiment run (e.g. `adapter-test`)
   - `--run_dir` -- name of the directory in which the experiment data will be saved (e.g. `runs/adapter-test`)
   - `--n_runs` -- number of experiment runs. If more than one, metric statistics will also be computed.
#### Weights and biases
   
We use weights and biases, and the root of the project is located at https://wandb.ai/we-robot/tar-project
To be able to log new runs, you must login to w&b. The easiest way to do this is:
   - before running the experiment, run `python -m wandb login`
   - go to https://wandb.ai and obtain your API key
   - paste API key to the command line
   - thats it, everything should work

#### Main experiments
    `python -m experiments.main <cli_args>
