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

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def generate_optimization_space():
    # Set a search range for hyperparameters
    paramSpace = {}
    paramSpace["torch.optim.SGD"] =     {"lr": hp.loguniform("lr", -12, 0)}
    paramSpace["torch.optim.Adagrad"] = {"lr": hp.loguniform("lr", -12, 0)}
    paramSpace["optim.AggMo"] = {
         "lr": hp.loguniform("lr", -12, 0),
         "weight_decay": hp.loguniform("weight_decay", -12, -2),
         "beta1": hp.uniform("beta1", 0.0, 1.0),
         "beta2": 1-hp.loguniform("beta2", -0.2, 0.0),
         "beta3": 1-hp.loguniform("beta3", -0.2, 0.0),
         }

    paramSpace["torch.optim.Rprop"] = {
        "lr": hp.loguniform("lr", -12, 0),
        "mum": hp.uniform("mum", 0.5, 1.0),
        "mup": hp.uniform("mup", 1.0, 1.5),
    }

    paramSpace["torch.optim.RMSprop"] = {
        "lr": hp.loguniform("lr", -12, 0), 
        "alpha": hp.loguniform("alpha", -0.2, 0.0)
    }
    
    adam_like = {
        "lr": hp.loguniform("lr", -12, 0),
        "beta1": hp.uniform("beta1", 0.0, 1.0),
        "beta2": 1-hp.loguniform("beta2", -0.2, 0.0),
    }

    paramSpace["torch.optim.Adam"] = adam_like
    paramSpace["torch.optim.Adamax"] = adam_like
    paramSpace["torch.optim.NAdam"] = adam_like
    paramSpace["torch.optim.RAdam"] = adam_like
    paramSpace["torch.optim.AMSgrad"] = adam_like
    
    paramSpace["optim.NovoGrad"] = adam_like
    paramSpace["optim.SWATS"] = adam_like
    paramSpace["optim.DiffGrad"] = adam_like
    paramSpace["optim.Yogi"] = adam_like
    paramSpace["optim.Lamb"] = adam_like
    paramSpace["optim.AdamP"] = adam_like

    paramSpace["torch.optim.SGDW"] = {
        "lr": hp.loguniform("lr", -12, 0),
        "weight_decay": hp.loguniform("weight_decay", -12, -2)
    }

    paramSpace["torch.optim.AdamW"] = {
        "lr": hp.loguniform("lr", -12, 0),
        "beta1": hp.uniform("beta1", 0.0, 1.0),
        "beta2": 1-hp.loguniform("beta2", -0.2, 0.0),
        "weight_decay": hp.loguniform("weight_decay", -12, 0)
    }

    paramSpace["optim.AdaMod"] = {
        "lr": hp.loguniform("lr", -12, 0),
        "beta1": hp.uniform("beta1", 0.0, 1.0),
        "beta2": 1-hp.loguniform("beta2", -0.2, 0.0),
        "weight_decay": hp.loguniform("weight_decay", -12, 0)
    }

    paramSpace["optim.MADGRAD"] = {
        "lr": hp.loguniform("lr", -12, 0),
        "weight_decay": hp.loguniform("weight_decay", -12, 0),
        "momentum": hp.loguniform("momentum", -12, 0),
    }

    paramSpace["optim.AdaBound"] = {
        "lr": hp.loguniform("lr", -12, 0),
        "beta1": hp.uniform("beta1", 0.0, 1.0),
        "beta2": 1-hp.loguniform("beta2", -0.2, 0.0),
        "final_lr": hp.loguniform("final_lr", -12, 0),
        "gamma": hp.loguniform("gamma", -12, 0),
    }

    paramSpace["optim.PID"] = {
        "lr": hp.loguniform("lr", -12, 0),
        "integral": hp.uniform("integral", 0.0, 10.0),
        "derivative": hp.uniform("derivative", 0.0, 10.0),
    }

    paramSpace["optim.QHAdam"] = {
        "lr": hp.loguniform("lr", -12, 0),
        "beta1": hp.uniform("beta1", 0.0, 1.0),
        "beta2": 1-hp.loguniform("beta2", -0.2, 0.0),
        "nus1": hp.uniform("nus1", 0.0, 1.0),
        "nus2": hp.uniform("nus2", 0.0, 1.0),   
    }
    return paramSpace


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

def Hyperparameters_optimization(Algorithm, space, Rosenbrock_B, nsamples=20, steps=100, hyperopt_steps=100):
    each = Algorithm
    nice_label = each.replace("optim.", "")
    nice_label = nice_label.replace("torch.", "")
    if os.path.isfile(f'data/{nice_label}_{Rosenbrock_B}_hypertrack.json'):
        print('Hyper Optimization was preempted')
        return
    else:
        print(f'Starting hyperparameters optimziation of {nice_label} with b={Rosenbrock_B}')

    # Refuse file save
    is_HPO = True

    # Rosenbrock function with floating parameter B
    def Rosenbrock_function(*x):
        res = 0.0
        for i in range(len(x) - 1):
            res += Rosenbrock_B * (x[i] - x[i + 1] ** 2.0) ** 2.0 + (1 - x[i + 1]) ** 2.0
        return res

    hyperparameters_loss = []

    def Rosenbrock_optimization(hyperparameters):
        output_data = {}
        aggregate_loss = []

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
            with open(f'data/{nice_label}_{Rosenbrock_B}_optimtrack.json', "w") as json_file:
                json.dump(output_data, json_file, indent=4)
        else:
            result = np.mean(aggregate_loss)
            hyperparameters_loss.append(result)
            return result
        
    # Perform the hyperparameters optimization
    best = fmin(Rosenbrock_optimization, space, algo=tpe.suggest, max_evals=hyperopt_steps)
    hyperparameter_dict = {}
    hyperparameter_dict["Best"] = best.copy()
    decrease_loss = []
    min_so_far = np.inf
    for each in hyperparameters_loss:
        min_so_far = min(min_so_far, each)
        decrease_loss.append(min_so_far)
    hyperparameter_dict["Track"] = decrease_loss.copy()
    with open(f'data/{nice_label}_{Rosenbrock_B}_hypertrack.json', "w") as json_file:
        json.dump(hyperparameter_dict, json_file, indent=4, cls=NpEncoder)

    # Save the result
    is_HPO = False
    Rosenbrock_optimization(best)

if __name__ == "__main__":
    folder_name = "data"

    # Check and create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created.")
    is_parallel = True
    optimization_space = generate_optimization_space()
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
    Rosenbrock_list = [1.0, 10.0, 100.0, 1000.0]

    task_list = []
    for Algorithm in algorithm_list:
        for Rosenbrock_parameter in Rosenbrock_list:
            parameters = (
                    Algorithm,
                    optimization_space[Algorithm], 
                    Rosenbrock_parameter, 
                    100, 
                    1000,
                    500
                )
            task_list.append(parameters)
            if not is_parallel:
                Hyperparameters_optimization(*parameters)
                
    if is_parallel:
        with Pool() as pool:
            # Use starmap to pass multiple arguments
            results = pool.starmap(Hyperparameters_optimization, task_list)
