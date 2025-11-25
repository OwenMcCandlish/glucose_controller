## Description:
This project aims to reproduce the work by Marchetti et al. (2025) on blood glucose control schemes. Their original study proposed a reinforcement learning approach that uses two different Proximal Policy Optimization (PPO) agents trained independently alongside a safety region for the artificial-pancreas problem. One agent is trained for hyperglycemic conditions, while the other is trained for hypoglycemic conditions. We re-implemented the Dual PPO architecture from scratch and evaluated it on 10 in silico adult patients using the UVA/Pandova simulator on randomized meal scenarios over a 5 day period. Our reproduction confirms the efficacy and data efficiency shown by the dual-agent approach in the paper. We achieved a mean Time-in-Range (TIR) of 66.10% ± 7.12 while the original paper achieved 69.30% ± 1.61. These results validate the original paper’s claim that separating control strategies for different glycemic regions enhances stability.


## How to Run
This project uses a command-line interface (CLI) to configure execution. The entrypoint for the CLI is *simulate.py*.

Command: `python3 simulate.py --model [model_name] [optional flags]`

Flags:
* --model [model_name]: Name of model to simulate. Searches through models/ for a file with a matching name.
* --train (optional): Forces the specified model to be retrained even if a previously trained model is available.
* --patient_name [patient_name] (optional: default="adult"): Name of patient to simulate model on
* --patient_num [patient_num] (optional: default="1"): Patient number to simulate model on

Available Patients: "adult", "child", "adolescent"
Available Patient Numbers: [1, 10]

Ex: `python3 simulate.py --model dual_ppo --train --patient_name adult --patient_num 7`
"Train a dual ppo model on adult patient number 7"


## Project Structure
```
.  
├── models  
│   ├── dual_ppo.py  
│   └── single_ppo.py  
├── stats  
│   ├─ adult#001.txt  
│   └── ...  
├── trained  
│   ├── dual_ppo  
│   │   ├── adult#001.zip  
│   │   └── ...  
│   └── single_ppo  
│       ├── adult#001.zip  
│       └── ...  
├── simulate.py  
├── run_all.sh  
└── run_experiments.sh  
```
    
models/:
Directory holding the source code for each individual model. To work with simulate.py, a model's source file must export a `create_model(env_factory, **kwargs)` function that takes in a function that creates a Gymnasium environment and returns a trained model, and optionally, a `reward_fun(observation)` function that takes an observation and returns a reward. 

models/dual_ppo.py:
Source code for the dual ppo controller model proposed by the reproduced paper.

models/single_ppo.py:
Source code for the single ppo controller model defined by the in the reproduced paper.

stats/:
Directory holding the results of a simulate.py simulation. A .txt file is generated depending on the name of the patient simulated.

trained/:
Directory holding trained models. A subdirectory is created for each model type. A model for each patient is stored with the name of the patient it's trained for in each subdirectory. 

simulate.py:
This is the main entry point for the project. It instantiates models, builds an environment, runs the primary validation simulation, and calculates runtime statistics. This script uses a command line interface described in the *How to Use* section.

run_all.sh:
Bash script that trains the dual_ppo model on all 10 adult patients.

run_experiments.sh:
Bash script that runs the duall_ppo model on all 10 adult patients 100 times and calculates aggregate results.


## Dependancies
* simglucose (https://github.com/jxx123/simglucose)
* Gymnasium (https://github.com/Farama-Foundation/Gymnasium)
* PyTorch (https://github.com/pytorch/pytorch)


## References
> Alessandro Marchetti, Daniele Sasso, Federico D’Antoni, Francesco Morandin, Maurizio Parton, Margherita Anna Grazia Matarrese, and Mario Merone. Deep reinforcement learning for type 1 diabetes: Dual ppo controller for personalized insulin management. In Computers in Biology and Medicine. Elsevier, 2025.

> Antonin Raffin, Ashley Hill, Adam Gleave, Anssi Kanervisto, Maximilian Ernestus, and Noah Dormann. Stable-baselines3: Reliable reinforcement learning implementations. Journal of Machine Learning Research, 22(268):1–8, 2021. URL http://jmlr.org/papers/v22/20-1364.html.

> Mark Towers, Ariel Kwiatkowski, Jordan Terry, John U Balis, Gianluca De Cola, Tristan Deleu, Manuel Goul˜ao, Andreas Kallinteris, Markus Krimmel, Arjun KG, et al. Gymnasium: A standard interface for reinforcement learning environments. arXiv preprint arXiv:2407.17032, 2024.

> Jinyu Xie. Simglucose v0.2.1. https://github.com/jxx123/simglucose, 2018. Ac-
cessed on: November-23-2025.

