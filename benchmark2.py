from multiprocessing import Process
from multiprocessing import Manager
import torch_optimizer as optim
import numpy as np
from copy import copy
import math
import torch
from hyperopt import tpe, hp, fmin
import statistics
import random
import json
from sklearn.linear_model import LinearRegression

# Control the seed of stochastic based optimization algorithms
torch.manual_seed(0)

# Number of optimization iterations
steps = 100
# Number of samples for initital conditions
nsamples = 20
# Epsilon for stop criteria
epsilon = 0.01

# The function save data instead of plotting
def postprocessing_into_file(arr, arr_x, name, label, logx=True):
    with open(name + "label_list.txt", "w") as fp:  # Pickling
        json.dump(label, fp)
    with open(name + "arg_list.txt", "w") as fp:  # Pickling
        json.dump(arr_x, fp)
    with open(name + "value_list.txt", "w") as fp:  # Pickling
        json.dump(arr, fp)


# The Nelder-Mead optimization algorithm
def nelder_mead(
    f, x_start, step=0.01, no_improve_thr=1e-2, no_improv_break=1000, max_iter=0
):
    # Constants
    dim = len(x_start)
    alpha = 1.0
    gamma = 2.0
    rho = -0.5
    sigma = 0.5
    all_prints = False

    # init
    prev_best = f(x_start)
    no_improv = 0
    res = [[x_start, prev_best]]

    for key in x_start:
        x = copy(x_start)
        x[key] = x[key] + step
        score = f(x)
        res.append([x, score])

    # simplex iter
    iters = 0
    while 1:
        # order
        res.sort(key=lambda x: x[1])
        best = res[0][1]
        # if all_prints:
        print(
            "Iteration: "
            + str(iters)
            + ". Best loss: "
            + str(best)
            + ". "
            + str(res[0][0])
        )
        # break after max_iter
        if max_iter and iters >= max_iter:
            return res[0]
        iters += 1

        if no_improv > no_improv_break:
            print("Method failed at a point:")
            return res[0]

        # break after no_improv_break iterations with no improvement
        # print('...best so far:', res[0][0])

        if best < prev_best - no_improve_thr:
            no_improv = 0
            prev_best = best
        else:
            no_improv += 1

        if best == 0.0:
            return res[0]

        num_of_more = 0
        # If the simplex is smaller then hypercube with 1e-6 side, then stop optimization.
        for key in x_start:
            local_list = []
            for tup in res:
                local_list.append(tup[0][key])
            if max(local_list) - min(local_list) > 1e-6:

                num_of_more += 1
        if num_of_more == 0:
            print("...best so far:", res[0])
            return res[0]

        # centroid
        x0 = copy(x_start)
        for key in x_start:
            x0[key] = 0.0
        for tup in res[:-1]:
            for k, v in tup[0].items():
                x0[k] += v / (len(res) - 1)

        # reflection
        xr = copy(x_start)
        for key in x_start:
            xr[key] = x0[key] + alpha * (x0[key] - res[-1][0][key])
        rscore = f(xr)
        if res[0][1] <= rscore < res[-2][1]:
            if all_prints:
                print("reflect")
            del res[-1]
            res.append([xr, rscore])
            continue

        # expansion
        if rscore < res[0][1]:
            xe = copy(x_start)
            for key in x_start:
                xe[key] = x0[key] + gamma * (x0[key] - res[-1][0][key])
            escore = f(xe)
            if all_prints:
                print("expansion")
            if escore < rscore:
                del res[-1]
                res.append([xe, escore])
                continue
            else:
                del res[-1]
                res.append([xr, rscore])
                continue

        # contraction
        if rscore >= res[-1][1]:
            xc = copy(x_start)
            for key in x_start:
                xc[key] = x0[key] + rho * (x0[key] - res[-1][0][key])
            cscore = f(xc)
            if cscore < res[-1][1]:
                if all_prints:
                    print("contraction inside")
                del res[-1]
                res.append([xc, cscore])
                continue
        else:
            xc = copy(x_start)
            for key in x_start:
                xc[key] = x0[key] - rho * (x0[key] - res[-1][0][key])
            cscore = f(xc)
            if cscore < res[-1][1]:
                if all_prints:
                    print("contraction outside")
                del res[-1]
                res.append([xc, cscore])
                continue

        # reduction
        x1 = res[0][0]
        nres = []
        if all_prints:
            print("reduction")
        for tup in res:
            redx = copy(x_start)
            for key in x_start:
                redx[key] = x1[key] + sigma * (tup[0][key] - x1[key])
            score = f(redx)
            nres.append([redx, score])
        res = nres


# Function for one independent process
def mp_func(opt, space, j, k, return_dict):
    print(str(j) + ": started")

    # Rosenbrock function with floating parameter k
    def rosen_convex(*x):
        res = 0.0
        for i in range(len(x) - 1):
            res += k * (x[i] - x[i + 1] ** 2.0) ** 2.0 + (1 - x[i + 1]) ** 2.0
        return res

    # Function of hyper parameter optimization
    def hyper(params):
        output_data = []
        # Collect some samples with randomized initial conditions
        for realization in range(nsamples):
            random.seed(realization)
            torch_x0 = [-1 - random.random(), 1.0 + random.random()]
            torch_params = torch.tensor(torch_x0, requires_grad=True)
            param_str = "([torch_params], "
            if "optim.AggMo" == opt:
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
            elif "torch.optim.Rprop" == opt:
                param_str = (
                    "([torch_params], lr="
                    + str(params["lr"])
                    + ", etas=("
                    + str(params["mum"])
                    + ", "
                    + str(params["mup"])
                    + ")"
                )
            elif "optim.QHAdam" == opt:
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
            if "torch.optim.AMSgrad" == opt:
                try:
                    optimizer = eval("torch.optim.Adam" + param_str + " amsgrad=True)")
                except:
                    loss = 9999.9
                    return loss
            elif "torch.optim.SGDW" == opt:
                try:
                    optimizer = eval("torch.optim.SGD" + param_str + ")")
                except:
                    loss = 9999.9
                    return loss
            else:
                try:
                    optimizer = eval(opt + param_str + ")")
                except:
                    import traceback
                    traceback.print_exc()
                    loss = np.inf
                    return loss
            #for _ in range(steps):
            prev = 10000.0
            while True:
                optimizer.zero_grad()
                loss = rosen_convex(*torch_params)
                loss.backward()
                optimizer.step()
                if abs(prev-loss) < epsilon:
                    break
                prev = loss
            if math.isnan(loss.item()):
                loss = 9999.9
                return loss
            else:
                output_data.append(loss.item())
        if output_data:
            return statistics.mean(output_data)
        else:
            return 9999.9

    # Hyper parameter optimization stage
    best = fmin(
        fn=hyper,  # Objective Function to optimize
        space=space,  # Hyperparameter's Search Space
        algo=tpe.suggest,  # Optimization algorithm
        max_evals=100,  # Number of optimization attempts
        rstate=np.random.default_rng(1),
    )
    best_dist = hyper(best)
    # If it fail try again with more attempts
    
    # Hyper parameters fine tunning
    best, _ = nelder_mead(hyper, best, step=0.01)
    return_dict[j] = hyper(best)
    return_dict[1000000 + j] = best


# Function for performance evaluation with fixed hyper parameters
def robust_test(opt, param, k):

    # Rosenbrock function with floating parameter k
    def rosen_convex(*x):
        res = 0.0
        for i in range(len(x) - 1):
            res += k * (x[i] - x[i + 1] ** 2.0) ** 2.0 + (1 - x[i + 1]) ** 2.0
        return res

    # Function for performance evaluation
    def hyper(params):
        output_data = []
        for realization in range(nsamples):
            random.seed(realization)
            torch_x0 = [-1 - random.random(), 1.0 + random.random()]
            torch_params = torch.tensor(torch_x0, requires_grad=True)
            param_str = "([torch_params], "
            if "optim.AggMo" == opt:
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
            elif "torch.optim.Rprop" == opt:
                param_str = (
                    "([torch_params], lr="
                    + str(params["lr"])
                    + ", etas=("
                    + str(params["mum"])
                    + ", "
                    + str(params["mup"])
                    + ")"
                )
            elif "optim.QHAdam" == opt:
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
            if "torch.optim.AMSgrad" == opt:
                try:
                    optimizer = eval("torch.optim.Adam" + param_str + " amsgrad=True)")
                except:
                    loss = 9999.9
                    return loss
            elif "torch.optim.SGDW" == opt:
                try:
                    optimizer = eval("torch.optim.SGD" + param_str + ")")
                except:
                    loss = 9999.9
                    return loss
            else:
                try:
                    optimizer = eval(opt + param_str + ")")
                except:
                    loss = 9999.9
                    return loss
            for _ in range(steps):
                optimizer.zero_grad()
                loss = rosen_convex(*torch_params)
                loss.backward()
                optimizer.step()
            if math.isnan(loss.item()):
                loss = 9999.9
                return loss
            else:
                output_data.append(loss.item())
        if output_data:
            return float(np.mean(np.array(output_data)))
            #return statistics.mean(output_data)
        else:
            return 9999.9

    return hyper(param)


# Function for computation distribution
def computational_conveuer(target, arg_list, n_cpu):

    procs = []
    previous_iterator = 0
    iterator = 0
    res_list = []
    param_path = []
    # Assign the task for the process
    while iterator < len(arg_list):
        manager = Manager()
        return_dict = manager.dict()
        for cpu in range(n_cpu):
            if iterator < len(arg_list):
                args_ret = (
                    arg_list[iterator][0],
                    arg_list[iterator][1],
                    iterator,
                    arg_list[iterator][2],
                    return_dict,
                )
                proc = Process(target=target, args=args_ret)
                procs.append(proc)
                proc.start()
                iterator += 1
        # complete the processes
        for proc in procs:
            proc.join()
        # Collect data from the processes
        for o in range(previous_iterator, iterator):
            value = return_dict[o]
            param = return_dict[1000000 + o]
            res_list.append(value)
            param_path.append(param)
            previous_iterator += 1
            if previous_iterator > 1000000:
                print("Heap is saturated!")
                exit
    return res_list, param_path


def log_slope(value_list, arg_list):
    log_value = []
    for each in value_list:
        if each == 0.0:
            log_value.append(1e-17)
        else:
            log_value.append(math.log(each, 10))
    log_arg = []
    for each in arg_list:
        log_arg.append([math.log(each, 10)])
    reg = LinearRegression().fit(log_arg, log_value)
    return reg.coef_[0]



if __name__ == "__main__":
    labels = []
    paramSpace = []
    if True:
        # Store considered algorithms
        
        labels.append("torch.optim.SGD")
        labels.append("torch.optim.Adagrad")
        labels.append("torch.optim.Rprop")
        labels.append("torch.optim.RMSprop")
        labels.append("torch.optim.Adam")
        labels.append("torch.optim.Adamax")
        labels.append("torch.optim.NAdam")
        labels.append("torch.optim.RAdam")
        labels.append("torch.optim.AMSgrad")
        labels.append("optim.PID")
        labels.append("optim.AdaBound")
        labels.append("optim.AdaMod")
        labels.append("torch.optim.AdamW")
        labels.append("torch.optim.SGDW")
        labels.append("optim.NovoGrad")
        labels.append("optim.AggMo")
        labels.append("optim.QHAdam")
        labels.append("optim.AdamP")
        labels.append("optim.SWATS")
        labels.append("optim.MADGRAD")
        labels.append("optim.DiffGrad")
        labels.append("optim.Yogi")
        labels.append("optim.Lamb")

        # Set a search range for hyperparameters
        paramSpace.append({"lr": hp.uniform("lr", 0.0, 1.0)})  # SGD
        paramSpace.append({"lr": hp.uniform("lr", 0.0, 1.0)})  # Adagrad
        paramSpace.append(
            {
                "lr": hp.uniform("lr", 0.0, 1.0),
                "mum": hp.uniform("mum", 0.5, 1.0),
                "mup": hp.uniform("mup", 1.0, 1.5),
            }
        )  # Rprop
        paramSpace.append(
            {"lr": hp.uniform("lr", 0.0, 0.5), "alpha": hp.uniform("alpha", 0.9, 1.0)}
        )  # RMSprop
        paramSpace.append(
            {
                "lr": hp.uniform("lr", 0.0, 0.5),
                "beta1": hp.uniform("beta1", 0.8, 1.0),
                "beta2": hp.uniform("beta2", 0.8, 1.0),
            }
        )  # Adam
        paramSpace.append(
            {
                "lr": hp.uniform("lr", 0.0, 0.5),
                "beta1": hp.uniform("beta1", 0.8, 1.0),
                "beta2": hp.uniform("beta2", 0.8, 1.0),
            }
        )  # AdaMax
        paramSpace.append(
            {
                "lr": hp.uniform("lr", 0.0, 1.5),
                "beta1": hp.uniform("beta1", 0.8, 1.0),
                "beta2": hp.uniform("beta2", 0.8, 1.0),
            }
        )  # NAdam
        paramSpace.append(
            {
                "lr": hp.uniform("lr", 0.0, 1.0),
                "beta1": hp.uniform("beta1", 0.5, 0.8),
                "beta2": hp.uniform("beta2", 0.5, 0.8),
            }
        )  # RAdam
        paramSpace.append(
            {
                "lr": hp.uniform("lr", 0.0, 1.5),
                "beta1": hp.uniform("beta1", 0.8, 1.0),
                "beta2": hp.uniform("beta2", 0.8, 1.0),
            }
        )  # AMSgrad
        paramSpace.append(
            {
                "lr": hp.uniform("lr", 0.0, 0.01),
                "integral": hp.uniform("integral", 0.2, 0.6),
                "derivative": hp.uniform("derivative", 0.8, 1.5),
            }
        )  # PID
        paramSpace.append(
            {
                "lr": hp.uniform("lr", 0.7, 1.5),
                "beta1": hp.uniform("beta1", 0.8, 1.0),
                "beta2": hp.uniform("beta2", 0.8, 1.0),
                "final_lr": hp.uniform("final_lr", 0.0, 0.02),
                "gamma": hp.uniform("gamma", 0.7, 0.9),
            }
        )  # AdaBound
        paramSpace.append(
            {
                "lr": hp.uniform("lr", 0.7, 0.9),
                "beta1": hp.uniform("beta1", 0.8, 1.0),
                "beta2": hp.uniform("beta2", 0.8, 1.0),
                "beta3": hp.uniform("beta3", 0.8, 1.0),
            }
        )  # AdaMod
        paramSpace.append(
            {
                "lr": hp.uniform("lr", 0.0, 0.3),
                "beta1": hp.uniform("beta1", 0.8, 1.0),
                "beta2": hp.uniform("beta2", 0.8, 1.0),
                "weight_decay": hp.uniform("weight_decay", 0.0, 0.03),
            }
        )  # AdamW
        paramSpace.append(
            {
                "lr": hp.uniform("lr", 0.0, 0.02),
                "weight_decay": hp.uniform("weight_decay", 0.7, 0.9),
            }
        )  # SGDW
        paramSpace.append(
            {
                "lr": hp.uniform("lr", 0.0, 0.04),
                "beta1": hp.uniform("beta1", 0.8, 1.0),
                "beta2": hp.uniform("beta2", 0.8, 1.0),
            }
        )  # Novograd
        paramSpace.append(
            {
                "lr": hp.uniform("lr", 0.0, 0.01),
                "beta1": hp.uniform("beta1", 0.5, 0.8),
                "beta2": hp.uniform("beta2", 0.5, 0.8),
                "beta3": hp.uniform("beta3", 0.8, 1.0),
                "weight_decay": hp.uniform("weight_decay", 0.0, 0.01),
            }
        )  # AggMo
        paramSpace.append(
            {
                "lr": hp.uniform("lr", 0.0, 0.3),
                "beta1": hp.uniform("beta1", 0.8, 1.0),
                "beta2": hp.uniform("beta2", 0.8, 1.0),
                "nus1": hp.uniform("nus1", 0.8, 1.0),
                "nus2": hp.uniform("nus2", 0.8, 1.0),
            }
        )  # QHAdam
        paramSpace.append(
            {
                "lr": hp.uniform("lr", 0.0, 2.0),
                "beta1": hp.uniform("beta1", 0.8, 1.0),
                "beta2": hp.uniform("beta2", 0.8, 1.0),
                "delta": hp.uniform("delta", 0.0, 1.0),
            }
        )  # AdamP
        paramSpace.append(
            {
                "lr": hp.uniform("lr", 0.0, 1.0),
                "beta1": hp.uniform("beta1", 0.8, 1.0),
                "beta2": hp.uniform("beta2", 0.8, 1.0),
            }
        )  # SWATS
        paramSpace.append(
            {
                "lr": hp.uniform("lr", 0.0, 0.1),
                "momentum": hp.uniform("momentum", 0.8, 1.0),
                "weight_decay": hp.uniform("weight_decay", 0.0, 0.03),
            }
        )  # MADGRAD
        paramSpace.append(
            {
                "lr": hp.uniform("lr", 0.25, 0.5),
                "beta1": hp.uniform("beta1", 0.8, 1.0),
                "beta2": hp.uniform("beta2", 0.8, 1.0),
            }
        )  # DiffGrad
        paramSpace.append(
            {
                "lr": hp.uniform("lr", 0.8, 1.2),
                "beta1": hp.uniform("beta1", 0.8, 1.0),
                "beta2": hp.uniform("beta2", 0.6, 0.9),
            }
        )  # Yogi
        paramSpace.append(
            {
                "lr": hp.uniform("lr", 0.0, 0.3),
                "beta1": hp.uniform("beta1", 0.8, 1.0),
                "beta2": hp.uniform("beta2", 0.6, 0.9),
            }
        )  # Lamb

    n = len(labels)
    # Set number of coefficient b of Rosenbrock function
    num_of_values = 4
    arr = [[] for _ in range(n)]
    path = [[] for _ in range(n)]
    arr_x = []
    arg_list = []
    full_var = True
    # Generate tasks
    for og in range(n):
        for jter in range(num_of_values):
            k = 10 ** jter  # Log scale parameter
            if og == 0:
                arr_x.append(k)
            arg_list.append((labels[og], paramSpace[og], k))
    # Compute in parallel
    out, param = computational_conveuer(mp_func, arg_list, 96)
    # store the data
    kter = 0
    for og in range(n):
        for jter in range(num_of_values):
            arr[og].append(out[kter])
            path[og].append(param[kter])
            kter += 1
    param_path = []
    hard_set = []

    postprocessing_into_file(
        arr, arr_x, "Rosenbrock2 function with hyper-optimization", labels
    )
    