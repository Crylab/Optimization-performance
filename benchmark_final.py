import torch_optimizer as optim
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from hyperopt import tpe, hp, fmin
import os.path
import os
from multiprocessing import Pool

def hyperparameters_to_string(params, Algorithm_name) -> str:
    '''
    This function returns the string optimization step command from 
    the hyperparameters dict and algorithm name for eval exeution.

    The problem here is that some of the algorithmn are 
    require a specific ways of hyperparameters providing. 
    All those special proceduares are included in this function and 
    incapsulate such complexity to make a process of optimization smooth and sound.
    '''

    param_str = "([torch_params], "
    if "optim.AggMo" == Algorithm_name:
        param_str = (
            "([torch_params], lr="
            + str(params["lr"])
            + ", weight_decay="
            + str(params["weight_decay"])
            + ", betas = ("
            + str(params["beta1"])
            + ", "
            + str(params["beta2"])
            + ", "
            + str(params["beta3"])
            + ")"
        )
    elif "torch.optim.Rprop" == Algorithm_name:
        param_str = (
            "([torch_params], lr="
            + str(params["lr"])
            + ", etas=("
            + str(params["mum"])
            + ", "
            + str(params["mup"])
            + ")"
        )
    elif "optim.QHAdam" == Algorithm_name:
        param_str = (
            "([torch_params], lr="
            + str(params["lr"])
            + ", betas=("
            + str(params["beta1"])
            + ", "
            + str(params["beta2"])
            + "), nus=("
            + str(params["nus1"])
            + ", "
            + str(params["nus2"])
            + ")"
        )
    else:
        if "beta1" in params and "beta2" in params:
            param_str = (
                "([torch_params], betas = ("
                + str(params["beta1"])
                + ", "
                + str(params["beta2"])
                + "),"
            )
        for key in params:
            if key != "beta1" and key != "beta2":
                param_str = param_str + key + "=" + str(params[key]) + ", "
    if "torch.optim.AMSgrad" == Algorithm_name:
            command = "torch.optim.Adam" + param_str + " amsgrad=True)"
    elif "torch.optim.SGDW" == Algorithm_name:
            command = "torch.optim.SGD" + param_str + ")"
    else:
            command = Algorithm_name + param_str + ")"
    return command

def Rosenbrock_optimization(Algorithm, hyperparameters, Rosenbrock_B=1.0, nsamples=100, steps=1000):
        is_HPO = False
        output_data = {}
        nice_label = Algorithm.replace("optim.", "").replace("torch.", "")
        aggregate_loss = []

        # Rosenbrock function with floating parameter B
        def Rosenbrock_function(*x):
            res = 0.0
            for i in range(len(x) - 1):
                res += Rosenbrock_B * (x[i] - x[i + 1] ** 2.0) ** 2.0 + (1 - x[i + 1]) ** 2.0
            return res

        # Collect some samples with randomized initial conditions
        for realization in range(nsamples if is_HPO else nsamples*10):
            np.random.seed(realization)
            # Randomized initial conditions
            torch_x0 = np.random.rand(2) * 2.0
            torch_params = torch.tensor(torch_x0, requires_grad=True)
            # Python code for optimizer with hyperparameters generation
            command = hyperparameters_to_string(hyperparameters, Algorithm)
            optimizer = eval(command)
            optimization_track = []
            optimization_loss = []

            # Launching optimization steps
            for _ in range(steps):
                optimizer.zero_grad()
                loss = Rosenbrock_function(*torch_params)
                optimization_track.append(torch_params.tolist())
                optimization_loss.append(loss.item())
                loss.backward()
                optimizer.step()
            aggregate_loss.append(sum(optimization_loss))
            output_data[f'track_{realization}'] = optimization_track.copy()
            output_data[f'loss_{realization}'] = optimization_loss.copy()
        
        if not is_HPO:
            with open(f'data_without/{nice_label}_{Rosenbrock_B}_without_hyper.json', "w") as json_file:
                json.dump(output_data, json_file, indent=4)

if __name__ == "__main__":
    folder_name = "data_without"

    # Check and create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created.")

    algorithm_list = [
        "torch.optim.SGD",
        "torch.optim.Adagrad",
        "optim.AggMo",
        "torch.optim.Rprop",
        "torch.optim.RMSprop",
        "torch.optim.Adam",
        "torch.optim.Adamax",
        "torch.optim.NAdam",
        "torch.optim.RAdam",
        "torch.optim.AMSgrad", 
        "optim.NovoGrad",
        "optim.SWATS",
        "optim.DiffGrad", 
        "optim.Yogi",
        "optim.Lamb",
        "optim.AdamP", 
        "torch.optim.SGDW", 
        "torch.optim.AdamW",
        "optim.AdaMod",
        "optim.MADGRAD",
        "optim.AdaBound", 
        "optim.PID",
        "optim.QHAdam",
    ]

    task_list = []
    for Rosenbrock_b in [1.0, 10.0, 100.0]:
        for Algorithm in algorithm_list:

            nice_label = Algorithm.replace("optim.", "").replace("torch.", "")
            file_name = nice_label + '_1000.0_hypertrack'

            with open("data/"+file_name+".json", "r") as json_file:
                data = json.load(json_file)
                parameters = (
                    Algorithm,
                    data["Best"],
                    Rosenbrock_b,
                )
                task_list.append(parameters)

    with Pool() as pool:
        # Use starmap to pass multiple arguments
        pool.starmap(Rosenbrock_optimization, task_list)
