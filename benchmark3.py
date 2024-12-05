import torch_optimizer as optim
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Control the seed of stochastic based optimization algorithms
torch.manual_seed(0)

# Number of optimization iterations
steps = 100
# Number of samples for initital conditions
nsamples = 3000
# Epsilon for stop criteria
epsilon = 0.00001

def rosenbrock1(*x):
    res = 0.0
    for i in range(len(x) - 1):
        res += (x[i] - x[i + 1] ** 2.0) ** 2.0 + (1 - x[i + 1]) ** 2.0
    return res

def rosenbrock10(*x):
    res = 0.0
    for i in range(len(x) - 1):
        res += 10 * (x[i] - x[i + 1] ** 2.0) ** 2.0 + (1 - x[i + 1]) ** 2.0
    return res

def rosenbrock100(*x):
    res = 0.0
    for i in range(len(x) - 1):
        res += 100 * (x[i] - x[i + 1] ** 2.0) ** 2.0 + (1 - x[i + 1]) ** 2.0
    return res

# Function for one independent process
def mp_func(opt, params, obj):

    output_data = {}
    
    # Collect some samples with randomized initial conditions
    for realization in range(nsamples):
        np.random.seed(realization)
        torch_x0 = np.random.rand(2) * 2.0
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
            optimizer = eval("torch.optim.Adam" + param_str + " amsgrad=True)")
        elif "torch.optim.SGDW" == opt:
            optimizer = eval("torch.optim.SGD" + param_str + ")")
        else:
            optimizer = eval(opt + param_str + ")")

        prev = np.inf
        optimization_path = []
        for _ in range(steps):
            optimizer.zero_grad()
            loss = obj(*torch_params)
            optimization_path.append(torch_params.tolist())
            loss.backward()
            optimizer.step()
            prev = loss
        output_data[realization] = optimization_path.copy()

    each = opt
    nice_label = each.replace("optim.", "")
    nice_label = nice_label.replace("torch.", "")
    file_name = f"{nice_label}"
    with open("data/"+file_name+".json", "w") as json_file:
        json.dump(output_data, json_file, indent=4)  # `indent=4` makes the JSON file more readable
    return file_name

    
def plot_points_from_json(file_name, obj):
    # Load JSON file
    with open("data/"+file_name+".json", "r") as json_file:
        data = json.load(json_file)

    x = np.linspace(0, 2, 500)  # Range for x
    y = np.linspace(0, 2, 500)  # Range for y
    X, Y = np.meshgrid(x, y)     # Create a 2D grid
    Z = obj(X, Y)         # Evaluate the function on the grid
    contour = plt.contour(X, Y, Z, levels=50, cmap="viridis", )
    plt.colorbar(contour, label="Function value",
                 norm=mcolors.LogNorm(vmin=Z.min() + 1e-3, vmax=Z.max())
                 ) 
    
    # Iterate through keys (e.g., "0", "1", ...) and plot points
    for key, points in data.items():
        # Extract x and y values
        x_vals = [point[0] for point in points]
        y_vals = [point[1] for point in points]
        
        # Plot the points
        plt.plot(x_vals, y_vals, label=f"Series {key}")
    
    # Customize the plot
    plt.title("X-Y Points from JSON")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim([0, 2])
    plt.ylim([0, 2])
    plt.axis('equal')
    plt.grid(True)
    plt.savefig("img/"+file_name+".pdf", format="pdf", dpi=300)


if __name__ == "__main__":

    result = mp_func("torch.optim.SGD", {"lr": 0.01}, rosenbrock1)
    print(result)
    plot_points_from_json(result, rosenbrock1)
    exit()

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

        

    