# Environment Setup

# Valid for mac, windows, linux

1. Install anaconda or miniconda.

2. Create a conda environment from the .yml files provided in `/environment` folder:
    - If you are running windows, use the Conda Prompt, on Mac or Linux you can just use the Terminal.
    - Use the command: `conda env create -f JY_practicum_env.yml`
    - This should create an environment named `JY_practicum_env`. 

3. Activate the conda environment:
    - Windows command: `activate JY_practicum_env` 
    - MacOS / Linux command: `conda activate JY_practicum_env`

For more references on conda environments, refer to [Conda Managing Environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or the [Conda Cheat Sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)