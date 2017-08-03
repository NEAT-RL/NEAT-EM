# NEAT-EM
NEAT + Expectation Maximisation

# Setup
First install python (v 3.5 is recommended for OpenAI libraries) and install conda or [miniconda](https://conda.io/docs/install/quick.html). Miniconda is recommended as its smaller and useful if you have space requirements.

Set up a conda environment by following the Project dependencies [repo](https://github.com/NEAT-RL/Project-Dependencies)

Install the organisation's custom [gym](https://github.com/NEAT-RL/gym) [gym-ple](https://github.com/NEAT-RL/gym-ple) libraries into the conda environment. These libraries extend openai gym. If you are looking on running gym-ple games, then you will need to install the PyGame-Learning-Environment from [here](https://github.com/NEAT-RL/PyGame-Learning-Environment).

# Running algorithm
The main file is *NEATEM.py*.
Experiment logs are written into log directory.

Run this file and if you want to save the std outputs for clarity use following command:

```
python NEATEM.py > output.log 2&>1
```


