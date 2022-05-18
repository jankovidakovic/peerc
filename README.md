# emotion-classification-using-transformers
We, robot

### Conda environment stuff
 - installing the environment: `conda env create --file environemtn.yaml`
 - activating: 
    - windows: `activate <env_name>`
    - linux: `source activate <env_name>`
 - adding new packages:
    1. install the package (preferably with conda, if the package is not available through conda, then install with pip)
    	- conda: `conda install -c <channel_name> <package_name>`
	   - specify channel name only if needed, its usually `conda-forge`, check package documentation for specific information
        - pip: `pip install <package_name>` (if first time adding pip package, run `conda install pip` first)
    2. add the package information to `environment.yaml`:
        - if installed using conda: just add into the list of dependencies
	- if installed using pip: add to list of pip dependencies
	- specify the package version, it insures that our code works forever and is not broken by package updates
 - conda cheat sheet: https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf
