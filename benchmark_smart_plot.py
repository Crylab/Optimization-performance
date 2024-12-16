import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import json
from sklearn.linear_model import LinearRegression
import math
import matplotlib.cm as cm
import numpy as np
import os
from matplotlib.ticker import LogLocator
from multiprocessing import Pool
from matplotlib.colors import LogNorm

viridis = plt.get_cmap()

def plot_horizontal_bar_chart(data, ax, compare_data = None, xlabel='X label', show_increase=True, low_labels = False, reverse=True):
    """
    Plots a horizontal bar chart on the provided AxesSubplot (ax) with a log-scaled x-axis.
    
    Parameters:
        data (dict): Dictionary where keys are categories (str) and values are corresponding numeric values.
        ax (AxesSubplot): Matplotlib AxesSubplot object to draw the chart on.
    """
    # Sort the data by values
    sorted_data = dict(sorted(data.items(), key=lambda item: item[1], reverse=reverse))
    
    # Generate positions and colors
    categories = list(sorted_data.keys())
    values = list(sorted_data.values())
    num_categories = len(categories)+2
    colors = cm.viridis(np.linspace(0.0, 1.0, num_categories))
    error_colors = cm.magma(np.linspace(0.5, 1.0, num_categories))

    color_smart = []
    error_color_smart = []
    half = int((len(categories)+1)/2)
    for i in range(half):
        color_smart.append(colors[i])
        color_smart.append(colors[i+half])
        error_color_smart.append(error_colors[i])
        error_color_smart.append(error_colors[i+half])

    
    if compare_data:
        # Plot horizontal bars
        ax.barh(categories, values, color=error_color_smart, height=1.0)
        
        error_values = []
        for ii, each in enumerate(categories):
            error_values.append(compare_data[each])
            diff = (data[each]-compare_data[each])/compare_data[each]
            if show_increase:
                if diff < 1.0 and low_labels:
                    ax.text(values[ii]*2 if diff > 0 else error_values[ii]*2, ii-0.25, f'{diff*100:.0f}%', ha='center')
                else:
                    ax.text(values[ii]*0.5 if diff < 11 else values[ii]*0.35, ii-0.25, f'×{diff:.2f}', ha='center')

        ax.barh(categories, error_values, color=color_smart, height=1.0)    
    else:
        ax.barh(categories, values, color=color_smart, height=1.0)
    

    # Set log scale for the x-axis
    ax.set_xscale("log")
    ax.set_ylim((-0.5, len(categories)-0.5))
    ax.set_xlabel(xlabel)
    ax.grid(True, linestyle='--', linewidth=0.7, color='gray', alpha=0.7)
 
def plot_portrait(file_name, ax, n_trajectories = 1000, name='', left=False, bottom=False, description=''):
    # Load JSON file
    
    with open("data/"+file_name+".json", "r") as json_file:
        data = json.load(json_file)

    # Iterate through keys (e.g., "0", "1", ...) and plot points
    i = 0
    for key, points in data.items():
        
        # Extract x and y values
        if "track" in key and i < n_trajectories:
            i += 1
            x_vals = [point[0] for point in points]
            y_vals = [point[1] for point in points]
    
            # Plot the points
            ax.plot(x_vals, y_vals, color=cm.viridis(i/n_trajectories), linewidth=1.0, alpha=0.25)
            # ax.plot(x_vals, y_vals, color=cm.viridis(np.random.random()), linewidth=1.0)
    
    # Customize the plot
    cross_size = 0.05
    low_cross = 1.0-cross_size
    up_cross = 1.0+cross_size
    path_effects=[pe.Stroke(linewidth=4, foreground='red'), pe.Normal()]
    ax.plot([low_cross, up_cross], [up_cross, low_cross], color='white', linewidth = 2, path_effects = path_effects)  # Horizontal line of the cross
    ax.plot([low_cross, up_cross], [low_cross, up_cross], color='white', linewidth = 2, path_effects = path_effects)  # Vertical line of the cross
    ticks = [0.0, 0.5, 1.0, 1.5, 2.0]
    tick_labels = ['0', '0.5', '1', '1.5', '2'] 
    ax.set_xticks(ticks)
    ax.axes.xaxis.set_ticklabels([])
    ax.set_yticks(ticks)
    ax.axes.yaxis.set_ticklabels([])
    if left:      
        ax.set_yticklabels(tick_labels)

    if bottom:
        ax.set_xticklabels(tick_labels)
    
    ax.set_xlim([0, 2])
    ax.set_ylim([0, 2])
    ax.set_aspect('equal', 'box')
    ax.text(
        0.05, 0.95,                   # Position (x, y) in axes coordinates
        name,                         # The text content
        fontsize=14,                  # Font size
        ha='left', va='top',         # Align the text
        transform=ax.transAxes,       # Use the current axes' coordinate system
        alpha=0.9,
    ).set_bbox(dict(facecolor='white', alpha=0.6, edgecolor='gray', boxstyle='round'))

    ax.text(
        0.95, 0.05,                   # Position (x, y) in axes coordinates
        description[0],                # The text content
        fontsize=14,                  # Font size
        ha='right', va='bottom',         # Align the text
        transform=ax.transAxes,       # Use the current axes' coordinate system
        alpha=0.9,
    ).set_bbox(dict(facecolor=description[1], alpha=0.6, edgecolor='gray', boxstyle='round'))

    ax.grid(True)

def plot_loss(file_name, ax, n_trajectories = 1000, name='', linthresh=10e-8):
    # Load JSON file
    with open("data/"+file_name+".json", "r") as json_file:
        data = json.load(json_file)

    # Iterate through keys (e.g., "0", "1", ...) and plot points
    i = 0
    max_point = -np.inf
    min_point = np.inf
    for key, points in data.items():
        
        # Extract x and y values
        if "loss" in key and i < n_trajectories and not any(point > 10**10 for point in points):
            i += 1    
            # Plot the points with consistent color
            ax.plot(points, color=cm.viridis(i/n_trajectories), linewidth=1.0, alpha=0.5)
            # ax.plot(points, color=cm.viridis(np.random.random()), linewidth=1.0, alpha=0.5)

            max_point = max(points) if max(points) > max_point else max_point
            min_point = min(points) if min(points) < max_point else min_point

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss value")
    ax.set_ylim((0, None))
    ax.set_xlim((0, 1000))
    ax.set_yscale('symlog', linthresh=linthresh)

    even_powers = range(int(np.log10(linthresh)), int(np.log10(max_point)))  # Generate even powers from 0 to -24
    step = 2
    while len(even_powers) > 10:
        even_powers = range(int(np.log10(linthresh)), int(np.log10(max_point)), step)  # Generate even powers from 0 to -24
        step += 1
    ticks = [10**p for p in even_powers]
    ticks.append(0)
    tick_labels = [f'$10^{{{p}}}$' for p in even_powers]
    tick_labels.append('0')
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels)

    text = ax.text(
        0.98, 0.95,                   # Position (x, y) in axes coordinates
        name,                         # The text content
        fontsize=14,                  # Font size
        ha='right', va='top',         # Align the text
        transform=ax.transAxes,       # Use the current axes' coordinate system
        color='black',
        alpha=1,
    )
    text.set_bbox(dict(facecolor='white', alpha=0.6, edgecolor='gray', boxstyle='round'))
    ax.grid(True)


def rosen_visualization():

    # Generate grid data for x and y
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-2, 2, 400)
    X, Y = np.meshgrid(x, y)

    # Set custom boundaries for the colormap
    vmin = 0.1  # Avoid log(0) by setting a minimum value greater than 0
    vmax = 2000

    # Create the figure and subplots (2 plots in one figure)
    fig, axes = plt.subplots(2, 2, subplot_kw={'projection': '3d'}, figsize=(15, 10))

    b_list = [1, 10, 100, 1000]

    # Define the Rosenbrock function
    def rosenbrock(x, y, b=100):
        return (1 - x)**2 + b * (y - x**2)**2

    for i in range(4):
        # Calculate Z values using the Rosenbrock function
        Z = rosenbrock(X, Y, b_list[i])
        vmin = 0.1
        vmax = np.max(Z)

        norm = LogNorm(vmin=vmin, vmax=vmax)

        smart_ax = axes[int(i/2), i%2]

        # Plot the first surface on the first axis
        surf = smart_ax.plot_surface(X, Y, Z, cmap='viridis', norm=norm)

        smart_ax.set_title(f'Rosenbrock function with b={b_list[i]} \n $f(x, y) = (1 - x)^2 + {b_list[i]}(y - x^2)^2$')
        smart_ax.set_xlabel('X')
        smart_ax.set_ylabel('Y')
        smart_ax.set_zlabel('f(x, y)')
        smart_ax.view_init(elev=30, azim=60)  # Adjust the view angle for this subplot
        fig.colorbar(surf, ax=smart_ax)

    # Display the plot
    plt.tight_layout()
    plt.savefig(f"img/rosen.pdf")
    print(f"Look at the picture: img/rosen.pdf")

if __name__ == "__main__":
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

    if False:

        fig, ax = plt.subplots(1, 4, figsize=(12, 6))

        # Start to generate charts        
        data_compare_with = {}
        rosen_list = ['1.0', '10.0', '100.0', '1000.0']
        for n_plot, b in enumerate(rosen_list):
            parallel_task = []
            for each in algorithm_list:
                nice_label = each.replace("optim.", "").replace("torch.", "")
                file_name = f'data/{nice_label}_{b}_optimtrack.json' 
                parallel_task.append(file_name)
            def parallel_acc(file_name):
                with open(file_name, "r") as json_file:
                    data = json.load(json_file)
                    accumulate = 0.0
                    total_number = 300
                    for i in range(300):
                        value = sum(data[f'loss_{i}'])
                        if not np.isnan(value):
                            accumulate += value
                        else:
                            total_number -= 1
                return accumulate/total_number
            with Pool() as pool:
                # Use starmap to pass multiple arguments
                results = pool.map(parallel_acc, parallel_task)

            plot_data = {}
            for i, each in enumerate(algorithm_list):
                nice_label = each.replace("optim.", "").replace("torch.", "")
                plot_data[nice_label] = results[i]

            # Plot the chart
            if n_plot == 0:
                plot_horizontal_bar_chart(plot_data, ax[n_plot], xlabel="Averaged integral metric of\n Rosenbrock parameter 1.0")
                data_compare_with = plot_data.copy()
            else:

                plot_horizontal_bar_chart(plot_data, ax[n_plot], compare_data=data_compare_with, xlabel=f'Averaged integral metric of\n Rosenbrock parameter {rosen_list[n_plot]}\n with respect to {rosen_list[n_plot-1]}')
                data_compare_with = plot_data.copy()

        # Axis failed in ax[0]
        ticks = [3.0, 4.0, 5.0, 6.0, 7.0, 10.0, 20.0]
        tick_labels = ['3', '4', '5', '', '7', '10', '20'] 
        ax[0].set_xticks(ticks)
        ax[0].axes.xaxis.set_ticklabels(tick_labels)

        # Adjust layout and show
        plt.tight_layout()
        plt.savefig("img/Smart_plot_1.pdf", format="pdf", dpi=300)
        plt.close()

    if False:

        fig, ax = plt.subplots(1, 4, figsize=(12, 6))

        # Start to generate charts  
        rosen_list = ['1.0', '10.0', '100.0', '1000.0']
        for n_plot, b in enumerate(rosen_list):
            ####################################
            ## Data with hyperparam optimization
            ####################################
            parallel_task = []
            for each in algorithm_list:
                nice_label = each.replace("optim.", "").replace("torch.", "")
                file_name = f'data/{nice_label}_{b}_optimtrack.json' 
                parallel_task.append(file_name)
            def parallel_acc(file_name):
                with open(file_name, "r") as json_file:
                    data = json.load(json_file)
                    accumulate = 0.0
                    total_number = 300
                    for i in range(300):
                        value = sum(data[f'loss_{i}'])
                        if not np.isnan(value):
                            accumulate += value
                        else:
                            total_number -= 1
                return accumulate/total_number
            with Pool() as pool:
                # Use starmap to pass multiple arguments
                results = pool.map(parallel_acc, parallel_task)

            plot_data = {}
            for i, each in enumerate(algorithm_list):
                nice_label = each.replace("optim.", "").replace("torch.", "")
                plot_data[nice_label] = results[i]

            ###############################
            ## Data without hyperparameter
            ###############################
            # Plot the chart
            if n_plot == 3:
                plot_horizontal_bar_chart(plot_data_without, ax[n_plot], xlabel="Averaged integral metric of\n Rosenbrock parameter 1000.0")
                continue
            parallel_task = []
            for each in algorithm_list:
                nice_label = each.replace("optim.", "").replace("torch.", "")
                file_name = f'data_without/{nice_label}_{b}_without_hyper.json' 
                parallel_task.append(file_name)
            def parallel_acc(file_name):
                with open(file_name, "r") as json_file:
                    data = json.load(json_file)
                    accumulate = 0.0
                    total_number = 300
                    for i in range(300):
                        value = sum(data[f'loss_{i}'])
                        if not np.isnan(value):
                            accumulate += value
                        else:
                            total_number -= 1
                return accumulate/total_number
            with Pool() as pool:
                # Use starmap to pass multiple arguments
                results = pool.map(parallel_acc, parallel_task)

            plot_data_without = {}
            for i, each in enumerate(algorithm_list):
                nice_label = each.replace("optim.", "").replace("torch.", "")
                plot_data_without[nice_label] = results[i]
   
            plot_horizontal_bar_chart(plot_data_without, ax[n_plot], compare_data=plot_data, xlabel=f'Averaged integral metric of\n Rosenbrock parameter {rosen_list[n_plot]}', low_labels = True)
                

        # Adjust layout and show
        plt.tight_layout()
        plt.savefig("img/Smart_plot_11.pdf", format="pdf", dpi=300)
        plt.close()

    if False:

        fig, ax = plt.subplots(1, 2, figsize=(6, 6))

        # Start to generate charts
        rosen_list = ['1000.0', '100.0', '10.0', '1.0']
        plot_data = {}
        plot_1000 = {}
        for n_plot, b in enumerate(rosen_list):
            for each in algorithm_list:
                nice_label = each.replace("optim.", "").replace("torch.", "")
                file_name = f'data/{nice_label}_{b}_hypertrack.json' 
                with open(file_name, "r") as json_file:
                    data = json.load(json_file)
                    if n_plot == 0:
                        plot_data[nice_label] = [data["Best"]["lr"]]
                        plot_1000[nice_label] = data["Best"]["lr"]
                    else:
                        plot_data[nice_label].append(data["Best"]["lr"])
        
        # Initialize the dictionaries
        min_dict = {}
        max_dict = {}
        relative_dict = {}

        # Populate the dictionaries
        for key, values in plot_data.items():
            min_val = min(values)
            max_val = max(values)
            min_dict[key] = min_val
            max_dict[key] = max_val
            relative_dict[key] = max_val/min_val

        plot_horizontal_bar_chart(plot_1000, ax[0],
                                  xlabel="Optimized learning rate for \n Rosenbrock function b=1000.0")
        plot_horizontal_bar_chart(relative_dict, ax[1],
                                  xlabel="Ratio of maximum and \nminimum values of \nlearning rate")

        # Adjust layout and show
        plt.tight_layout()
        plt.savefig("img/Smart_plot_2.pdf", format="pdf", dpi=300)
        plt.close()
   
    if False:
        fig, ax = plt.subplots(4, 1, figsize=(6, 12))
        # n_traj = 100
        plot_loss("SGD_1.0_optimtrack", ax[0], name="Asymp. convergence: 100 SGD real., b=1", n_trajectories=100, linthresh=10e-16)
        plot_loss("Adagrad_1.0_optimtrack", ax[1], name="Init. cond. dep.: 500 AdaGrad real., b=1", n_trajectories=500, linthresh=10e-25)
        plot_loss("Adam_1.0_optimtrack", ax[2], name="Stable oscillations: 5 Adam real., b=1", n_trajectories=5, linthresh=10e-6)
        plot_loss("Yogi_100.0_optimtrack", ax[3], name="Mixed behavior: 500 Yogi real., b=100", n_trajectories=500, linthresh=10e-5)

        plt.tight_layout(pad=0.1)
        plt.savefig("img/Smart_plot_3.pdf")
        plt.close()

    if False:
        fig, ax = plt.subplots(3, 2, figsize=(8, 12))
        n_traj = 200
        plot_portrait("AdaBound_1.0_optimtrack", ax[0, 0], name="AdaBound: b=1.0", n_trajectories=n_traj)
        plot_portrait("AdaBound_100.0_optimtrack", ax[1, 0], name="AdaBound: b=100.0", n_trajectories=n_traj)
        plot_portrait("AdaBound_1000.0_optimtrack", ax[2, 0], name="AdaBound: b=1000.0", n_trajectories=n_traj)

        plot_portrait("Adam_1.0_optimtrack", ax[0, 1], name="Adam: b=1.0", n_trajectories=n_traj)
        plot_portrait("Adam_100.0_optimtrack", ax[1, 1], name="Adam: b=100.0", n_trajectories=n_traj)
        plot_portrait("Adam_1000.0_optimtrack", ax[2, 1], name="Adam: b=1000.0", n_trajectories=n_traj)

        plt.tight_layout(pad=0.1)
        plt.savefig("img/Smart_plot_4.png", format="png", dpi=300)
        plt.close()

    if False:
        fig, ax = plt.subplots(3, 2, figsize=(8, 12))

        plot_portrait("MADGRAD_1.0_optimtrack", ax[0, 0], name="MADGRAD: b=1.0")
        plot_portrait("MADGRAD_10.0_optimtrack", ax[1, 0], name="MADGRAD: b=10.0")
        plot_portrait("MADGRAD_100.0_optimtrack", ax[2, 0], name="MADGRAD: b=100.0")

        plot_portrait("QHAdam_1.0_optimtrack", ax[0, 1], name="QHAdam: b=1.0")
        plot_portrait("QHAdam_10.0_optimtrack", ax[1, 1], name="QHAdam: b=10.0")
        plot_portrait("QHAdam_100.0_optimtrack", ax[2, 1], name="QHAdam: b=100.0")

        plt.tight_layout(pad=0.1)
        plt.savefig("img/Smart_plot_5.png", format="png", dpi=300)
        plt.close()

    if False:
        i=0
        j=0
        for algorithm in algorithm_list:
            if i == 0:
                fig, ax = plt.subplots(4, 3, figsize=(9, 12))
            nice_label = algorithm.replace("optim.", "").replace("torch.", "")

            plot_portrait(f'{nice_label}_1.0_optimtrack', ax[0, i], name=f'{nice_label}: b=1.0')
            plot_portrait(f'{nice_label}_10.0_optimtrack', ax[1, i], name=f'{nice_label}: b=10.0')
            plot_portrait(f'{nice_label}_100.0_optimtrack', ax[2, i], name=f'{nice_label}: b=100.0')
            plot_portrait(f'{nice_label}_1000.0_optimtrack', ax[3, i], name=f'{nice_label}: b=1000.0')
            if i == 2:
                plt.tight_layout(pad=0.1)
                plt.savefig(f'img/APNDX_{j}.png', format="png", dpi=300)
                plt.close()
                i = 0
                j+=1
            else:
                i+=1
        plt.tight_layout(pad=0.1)
        plt.savefig(f'img/APNDX_{j}.png', format="png", dpi=300)
        plt.close()

    if True:
        learning_rate = 0.001
        dict_acc = {}
        dict_auc = {}
        for Algorithm in algorithm_list:
            if Algorithm == "torch.optim.SGDW": continue
            if "torch.optim.AMSgrad" == Algorithm:
                result = "torch.optim.Adam(model.parameters(), lr=" + str(learning_rate) + ", amsgrad=True)"
            else:
                result = Algorithm + "(model.parameters(), lr=" + str(learning_rate) + ")"
            with open(f'data_ResNet/ResNet_{result}.json', "r") as json_file:
                data = json.load(json_file)
                nice_label = Algorithm.replace("optim.", "").replace("torch.", "")
                dict_acc[nice_label] = data["Total_accuracy"]
                dict_auc[nice_label] = sum(data["Optimization_path"])/100

        ### PLOTTING
        fig, ax = plt.subplots(1, 2, figsize=(6, 6))
        plot_horizontal_bar_chart(dict_acc, ax[0],
                                  xlabel="Recognition accuracy, %",
                                  reverse=False)
        plot_horizontal_bar_chart(dict_auc, ax[1],
                                  xlabel="Averaged integral metric \nof ResNet-18 training")
        # Adjust layout and show
        ax[0].set_xscale("linear")
        ax[1].set_xscale("linear")
        ax[0].set_xlim((50, 90))
        plt.tight_layout()
        plt.savefig("img/Smart_plot_6.pdf", format="pdf", dpi=300)
        plt.close()
        
    if False:
        fig, ax = plt.subplots(5, 4, figsize=(12, 15))
        n_traj = 200
        A = ('Asymptotic', 'tab:orange')
        M = ('Mixed', 'tab:green')
        O = ('Oscillating', 'tab:purple')
        D = ('Dependent', 'tab:pink')
        labels = {
            'AdaBound: b=1':A,
            'AdaBound: b=10':A,
            'AdaBound: b=100':A,
            'AdaBound: b=1000':A,

            'QHAdam: b=1':M,
            'QHAdam: b=10':O,
            'QHAdam: b=100':A,
            'QHAdam: b=1000':A,

            'MADGRAD: b=1':M,
            'MADGRAD: b=10':M,
            'MADGRAD: b=100':M,
            'MADGRAD: b=1000':M,

            'Adam: b=1':O,
            'Adam: b=10':O,
            'Adam: b=100':O,
            'Adam: b=1000':O,

            'Yogi: b=1':D,
            'Yogi: b=10':D,
            'Yogi: b=100':M,
            'Yogi: b=1000':M,
        }
        for i in range(4):
            for j, Algorithm in enumerate(['AdaBound', 'Yogi', 'QHAdam', 'MADGRAD', 'Adam']):
                plot_portrait(
                    f'{Algorithm}_{10**i}.0_optimtrack', 
                    ax[j, i], 
                    name=f'{Algorithm}: b={10**i}', 
                    n_trajectories=n_traj,
                    left = True if i == 0 else False,
                    bottom = True if j == 4 else False,         
                    description=labels[f'{Algorithm}: b={10**i}']           
                )

        plt.tight_layout(pad=0.1)
        plt.savefig("img/Smart_plot_7.pdf", dpi=300)
        plt.close()
