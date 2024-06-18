import matplotlib.pyplot as plt
import json
from sklearn.linear_model import LinearRegression
import math

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
    
    

if __name__ == "__main__":
    chart_with_optimization()
    chart_without_optimization()
    chart_slope()
    chart_bar()
    chart_robust()
    
    
    
    