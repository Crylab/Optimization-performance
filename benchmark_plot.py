import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import json
from sklearn.linear_model import LinearRegression
import math
import matplotlib.cm as cm
import numpy as np
import os
from matplotlib.ticker import LogLocator

viridis = plt.get_cmap()

def chart_with_optimization():
    with open("Rosenbrock function with hyper-optimizationarg_list.txt", 'r') as f:
        argument = json.load(f)
    with open("Rosenbrock function with hyper-optimizationlabel_list.txt", 'r') as f:
        label = json.load(f)
    with open("Rosenbrock function with hyper-optimizationvalue_list.txt", 'r') as f:
        value = json.load(f)
        
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    
    for each in range(len(label)):
        nice_label = label[each].replace("optim.", "")
        nice_label = nice_label.replace("torch.", "")
        plt_counter = each // 6
            
        obj = ax[0 if plt_counter < 2 else 1, plt_counter % 2]
        
        if plt_counter % 2 == 1:
            obj.yaxis.tick_right()
            obj.yaxis.set_label_position('right')
        obj.plot(argument, value[each], label=nice_label, color=viridis(float(each%6/6)))
        obj.grid(True)
        obj.set_yscale("symlog", linthresh=1e-16)
        obj.set_xscale("log")
        obj.set_ylabel("Loss")
        obj.set_ylim(0, 1)
        obj.set_xlim(argument[0], argument[-1])
        if each > 12:
            obj.set_xlabel("Rosenbrock parameter b")
        obj.legend()
        
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.suptitle("Performance on Rosnbrock function with hyperparameters optimization")
    fig.subplots_adjust(wspace=0.005, hspace=0.1)
    plt.savefig(f"img/Chart_with_opt.pdf")
    print(f"Look at the picture: img/Chart_with_opt.pdf")

def chart_without_optimization():
    with open("Rosenbrock function without hyper-optimizationarg_list.txt", 'r') as f:
        argument = json.load(f)
    with open("Rosenbrock function without hyper-optimizationlabel_list.txt", 'r') as f:
        label = json.load(f)
    with open("Rosenbrock function without hyper-optimizationvalue_list.txt", 'r') as f:
        value = json.load(f)
        
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    
    for each in range(len(label)):
        nice_label = label[each].replace("optim.", "")
        nice_label = nice_label.replace("torch.", "")
        plt_counter = each // 6
            
        obj = ax[0 if plt_counter < 2 else 1, plt_counter % 2]
        
        if plt_counter % 2 == 1:
            obj.yaxis.tick_right()
            obj.yaxis.set_label_position('right')
        obj.plot(argument, value[each], label=nice_label, color=viridis(float(each%6/6)))
        obj.grid(True)
        obj.set_yscale("symlog", linthresh=1e-6)
        obj.set_xscale("log")
        obj.set_ylabel("Loss")
        obj.set_ylim(0, 15)
        obj.set_xlim(argument[0], argument[-1])
        if each > 12:
            obj.set_xlabel("Rosenbrock parameter b")
        obj.legend()
        
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.suptitle("Performance on Rosenbrock function without hyperparameters optimization")
    fig.subplots_adjust(wspace=0.005, hspace=0.1)
    plt.savefig(f"img/Chart_without_opt.pdf")
    print(f"Look at the picture: img/Chart_without_opt.pdf")

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

def chart_slope():
    with open("Rosenbrock function with hyper-optimizationarg_list.txt", 'r') as f:
        arg = json.load(f)
    with open("Rosenbrock function with hyper-optimizationlabel_list.txt", 'r') as f:
        label = json.load(f)
    with open("Rosenbrock function with hyper-optimizationvalue_list.txt", 'r') as f:
        value = json.load(f)
    
    slope_list = []
    
    for each in value:
        slope_list.append(log_slope(each, arg))
    
    fig, ax = plt.subplots()
    value_arr = [x[-1] for x in value]
    label_arr = []
    
    for i in range(len(label)):
        nice_label = label[i].replace("optim.", "")
        nice_label = nice_label.replace("torch.", "")
        label_arr.append(nice_label)
    
    
    ax.scatter(slope_list, value_arr)
    
    for i,txt in enumerate(label_arr):
        vertical_shift = 1.1
        if txt in ["SGDW"]:
            vertical_shift = 0.75
        if txt in ["SWATS", "QHAdam", "SGD"]:
            vertical_shift = 0.9
        ax.annotate(txt, (slope_list[i], value_arr[i]*vertical_shift))
        
    #plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.suptitle("Algorithms map: overall performace vs performance decline rate")
    ax.set_xlabel("Performance slope coeffitient in Rosenbrick function")
    ax.set_ylabel("Loss at b=100 (overall performance)")
    ax.grid(True)
    ax.set_ylim(0.03,1.2)
    ax.set_xlim(0.5, 9)
    ax.set_yscale("symlog", linthresh=1e-1)
    plt.savefig(f"img/Slope.pdf")
    print(f"Look at the picture: img/Slope.pdf")

def chart_robust():
    with open("Rosenbrock function with hyper-optimizationarg_list.txt", 'r') as f:
        arg = json.load(f)
    with open("Rosenbrock function with hyper-optimizationlabel_list.txt", 'r') as f:
        label = json.load(f)
    with open("Rosenbrock function with hyper-optimizationvalue_list.txt", 'r') as f:
        value = json.load(f)
    
    slope_list_with = []
    
    for each in value:
        slope_list_with.append(log_slope(each, arg))
        
    del value
    with open("Rosenbrock function without hyper-optimizationvalue_list.txt", 'r') as f:
        value = json.load(f)
    
    slope_list_without = []
    
    for each in value:
        slope_list_without.append(log_slope(each, arg))
        
    diff = []
    for i in range(len(label)):
        diff.append((slope_list_with[i] - slope_list_without[i])/slope_list_with[i])
        
        
    with open("ParamPath.txt", 'r') as f:
        param = json.load(f)
                
    range_list = []
    for each in param:
        max_lr = 0
        min_lr = 1e9
        for every in each:
            if every['lr'] > max_lr:
                max_lr = every['lr']
            if every['lr'] < min_lr:
                min_lr = every['lr']
        range_list.append((max_lr-min_lr))#/max_lr)    
    
    
    
    fig, ax = plt.subplots()
    label_arr = []
    
    for i in range(len(label)):
        nice_label = label[i].replace("optim.", "")
        nice_label = nice_label.replace("torch.", "")
        label_arr.append(nice_label)
    
    
    ax.scatter(diff, range_list)
    
    for i,txt in enumerate(label_arr):
        vertical_shift = 1.1
        if txt in ["SGDW", "QHAdam", "Adamax"]:
            vertical_shift = 0.75
        if txt in ["SGD", "NAdam", "AdamW"]:
            vertical_shift = 0.9
        if txt == "AdaMod":
            ax.annotate(txt, (diff[i]-0.07, range_list[i]))
            continue
        ax.annotate(txt, (diff[i], range_list[i]*vertical_shift))
        
    plt.tight_layout(rect=[0.05, 0.05, 1.0, 0.97])
    fig.suptitle("Robust comparison: Slope relative difference vs Learning rate range")
    ax.set_xlabel("Slope relative difference between performance \n with and without hyperparameters optimization")
    ax.set_ylabel("Learning rate range")
    ax.grid(True)
    ax.set_ylim(0.0, 5)
    ax.set_xlim(0.25, 1.4)
    ax.set_yscale("symlog", linthresh=1e-1)
    plt.savefig(f"img/robust.pdf")
    print(f"Look at the picture: img/robust.pdf")

def chart_bar():
    with open("ParamPath.txt", 'r') as f:
        param = json.load(f)
        
    with open("Rosenbrock function with hyper-optimizationlabel_list.txt", 'r') as f:
        label = json.load(f)
        
    range_list = []
    for each in param:
        max_lr = 0
        min_lr = 1e9
        for every in each:
            if every['lr'] > max_lr:
                max_lr = every['lr']
            if every['lr'] < min_lr:
                min_lr = every['lr']
        range_list.append((max_lr-min_lr))#/max_lr)
        
    label_arr = []    
    for i in range(len(label)):
        nice_label = label[i].replace("optim.", "")
        nice_label = nice_label.replace("torch.", "")
        label_arr.append(nice_label)
        
        
    # Pair the labels and values together and sort them by values
    paired_list = sorted(zip(range_list, label_arr))
    sorted_values, sorted_labels = zip(*paired_list)
        
    # Create a bar chart
    plt.figure(figsize=(7, 4))
    plt.bar(sorted_labels, sorted_values, color='skyblue')
    
    # Add title and labels
    plt.title('Learning rate range')
    plt.ylabel('Learning rate range')
    plt.xticks(rotation=90)
    plt.grid(axis='y')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(f"img/Bar.pdf")
    print(f"Look at the picture: img/Bar.pdf")
    
def plot_points_from_json(file_name):
    # Load JSON file
    plt.figure(figsize=(8, 6)) 
    size = np.inf
    with open("data/"+file_name+".json", "r") as json_file:
        data = json.load(json_file)

    # Iterate through keys (e.g., "0", "1", ...) and plot points
    i = 0
    for key, points in data.items():
        
        # Extract x and y values
        if "track" in key and i < size:
            i += 1
            x_vals = [point[0] for point in points]
            y_vals = [point[1] for point in points]
    
            # Plot the points
            plt.plot(x_vals, y_vals, color=cm.viridis(np.random.rand()))
    
    # Customize the plot
    cross_size = 0.05
    low_cross = 1.0-cross_size
    up_cross = 1.0+cross_size
    path_effects=[pe.Stroke(linewidth=4, foreground='red'), pe.Normal()]
    plt.plot([low_cross, up_cross], [up_cross, low_cross], color='white', linewidth = 2, path_effects = path_effects)  # Horizontal line of the cross
    plt.plot([low_cross, up_cross], [low_cross, up_cross], color='white', linewidth = 2, path_effects = path_effects)  # Vertical line of the cross

    plt.title(f'Phase portrait of {file_name.split("_")[0]}')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim([0, 2])
    plt.ylim([0, 2])
    plt.axis('equal')
    plt.grid(True)
    plt.savefig("img/"+file_name+".pdf", format="pdf", dpi=300)

def plot_points_loss(file_name):
    # Load JSON file
    plt.figure(figsize=(8, 6)) 
    size = np.inf
    with open("data/"+file_name+".json", "r") as json_file:
        data = json.load(json_file)

    # Iterate through keys (e.g., "0", "1", ...) and plot points
    i = 0
    for key, points in data.items():
        
        # Extract x and y values
        if "loss" in key and i < size:
            i += 1    
            # Plot the points
            plt.plot(points, color=cm.viridis(np.random.rand()), )
    
    plt.title(f'Loss ')
    plt.xlabel("Epochs")
    plt.ylabel("Loss value")
    plt.yscale('log')
    plt.grid(True)
    plt.savefig("img/"+file_name+"_loss.pdf", format="pdf", dpi=300)
    plt.close()

def plot_hyperoptimization(algo_names):
    plt.figure(figsize=(8, 6)) 
    for i, each in enumerate(algo_names):
        file_name = f'{each}_1.0_hypertrack'
        try:
            with open("data/"+file_name+".json", "r") as json_file:
                data = json.load(json_file)
            plt.plot(data["Track"], color=cm.viridis(float(i/len(algo_names))), label=each)
        except:
            continue
        
    plt.title(f'Hyperparameter optimization dynamics')
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True)
    plt.legend()
    plt.savefig("img/hypertracks.pdf", format="pdf", dpi=300)
    plt.close()

def resnet_plot():
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
        "torch.optim.AdamW",
        "optim.AdaMod",
        "optim.MADGRAD",
        "optim.AdaBound", 
        "optim.PID",
        "optim.QHAdam",
    ]

    task_list = []
    for Algorithm in algorithm_list:
        for learning_rate in [10**i for i in range(-3, -2)]:
            if "torch.optim.AMSgrad" == Algorithm:
                result = "torch.optim.Adam(model.parameters(), lr=" + str(learning_rate) + ", amsgrad=True)"
            else:
                result = Algorithm + "(model.parameters(), lr=" + str(learning_rate) + ")"
            task_list.append(result)
    dict_acc = {}
    for i, each in enumerate(task_list):
        with open(f'data_ResNet/ResNet_{each}.json', "r") as json_file:
            data = json.load(json_file)
            nice_label = algorithm_list[i].replace("optim.", "")
            nice_label = nice_label.replace("torch.", "")
            dict_acc[nice_label] = data["Total_accuracy"]


    colors = plt.cm.viridis(np.linspace(0.0, 1.0, len(algorithm_list) +1))
    color_smart = []
    half = int((len(algorithm_list)+1)/2)
    for i in range(half):
        color_smart.append(colors[i])
        color_smart.append(colors[i+half])

    # Sort the dictionary by values
    sorted_data = dict(sorted(dict_acc.items(), key=lambda item: item[1]))

    # Extract keys and values
    categories = list(sorted_data.keys())
    values = list(sorted_data.values())

    # Create the horizontal bar chart
    plt.figure(figsize=(6, 10))
    bar_width=1.0
    bars = plt.barh(categories, values, log=True, color=color_smart, height=bar_width)

    # Add value annotations to the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height() / 2, f'{width}%', va='center', ha='left')

    # Labeling the chart
    plt.title('ResNet-18 model training on CIFAR-10 dataset')
    plt.xlabel('Recognition accuracy, %')
    plt.xlim([52, 90])
    plt.ylabel('Optimization algorithms')

    
    # Set major ticks at specific points
    major_ticks = range(50, 90, 5)
    plt.gca().xaxis.set_major_locator(LogLocator(base=10.0, subs=[], numticks=len(major_ticks)))
    # Set minor ticks for finer granularity
    minor_ticks = range(50, 90, 5)
    plt.gca().xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=len(minor_ticks)))
    # Enable grid
    plt.grid(which='minor', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)  # Minor grid

    plt.tight_layout()

    # Show the plot
    plt.savefig("img/Resnet_bar.pdf", format="pdf", dpi=300)
    plt.close()

if __name__ == "__main__":
    #chart_with_optimization()
    #chart_without_optimization()
    #chart_slope()
    #chart_bar()
    #chart_robust()
    resnet_plot()
    exit()
    folder_name = "img"

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
    another_list = []
    for each in algorithm_list:
        nice_label = each.replace("optim.", "")
        nice_label = nice_label.replace("torch.", "")
        another_list.append(nice_label)
        file_name = f'{nice_label}_1.0_optimtrack'    
        plot_points_from_json(file_name)
        try:
            plot_points_loss(file_name)
        except:
            print(f'File {file_name} has failed')
    plot_hyperoptimization(another_list)
  