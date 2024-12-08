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

viridis = plt.get_cmap()

def plot_horizontal_bar_chart(data, ax, compare_data = None, xlabel='X label', show_increase=True, low_labels = False):
    """
    Plots a horizontal bar chart on the provided AxesSubplot (ax) with a log-scaled x-axis.
    
    Parameters:
        data (dict): Dictionary where keys are categories (str) and values are corresponding numeric values.
        ax (AxesSubplot): Matplotlib AxesSubplot object to draw the chart on.
    """
    # Sort the data by values
    sorted_data = dict(sorted(data.items(), key=lambda item: item[1], reverse=True))
    
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
 
def plot_portrait(file_name, ax, n_trajectories = 1000):
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
            ax.plot(x_vals, y_vals, color=cm.viridis(i/n_trajectories), linewidth=1.0)
    
    # Customize the plot
    cross_size = 0.05
    low_cross = 1.0-cross_size
    up_cross = 1.0+cross_size
    path_effects=[pe.Stroke(linewidth=4, foreground='red'), pe.Normal()]
    ax.plot([low_cross, up_cross], [up_cross, low_cross], color='white', linewidth = 2, path_effects = path_effects)  # Horizontal line of the cross
    ax.plot([low_cross, up_cross], [low_cross, up_cross], color='white', linewidth = 2, path_effects = path_effects)  # Vertical line of the cross

    #ax.set_xlabel("X")
    #ax.set_ylabel("Y")
    ax.set_xlim([0, 2])
    ax.set_ylim([0, 2])
    ax.set_aspect('equal', 'box')
    ax.grid(True)

def plot_loss(file_name, ax, n_trajectories = 1000):
    # Load JSON file
    with open("data/"+file_name+".json", "r") as json_file:
        data = json.load(json_file)

    # Iterate through keys (e.g., "0", "1", ...) and plot points
    i = 0
    for key, points in data.items():
        
        # Extract x and y values
        if "loss" in key and i < n_trajectories and not any(point > 10**10 for point in points):
            i += 1    
            # Plot the points
            ax.plot(points, color=cm.viridis(i/n_trajectories), linewidth=1.0)
    
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss value")
    ax.set_yscale('log')
    ax.grid(True)

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

    if True:

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
        for n_plot, b in enumerate(rosen_list):
            for each in algorithm_list:
                nice_label = each.replace("optim.", "").replace("torch.", "")
                file_name = f'data/{nice_label}_{b}_hypertrack.json' 
                with open(file_name, "r") as json_file:
                    data = json.load(json_file)
                    if n_plot == 0:
                        plot_data[nice_label] = [data["Best"]["lr"]]
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

        plot_horizontal_bar_chart(max_dict, ax[0], compare_data=min_dict, show_increase=False,
                                  xlabel="Optimized learning rate: \nFrom minimal (shades of green) \nto maximum (shades of red)")
        plot_horizontal_bar_chart(relative_dict, ax[1],
                                  xlabel="Ratio of maximum and \nminimum values of \nlearning rate")

        # Adjust layout and show
        plt.tight_layout()
        plt.savefig("img/Smart_plot_2.pdf", format="pdf", dpi=300)
        plt.close()

    if False:

        fig, ax = plt.subplots(figsize=(12, 6))
        # ########################
        # Performance over b=1000     
        # ##########################   
        parallel_task = []
        for each in algorithm_list:
            nice_label = each.replace("optim.", "").replace("torch.", "")
            file_name = f'data/{nice_label}_1000.0_optimtrack.json'
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

        performance_data = {}
        for i, each in enumerate(algorithm_list):
            nice_label = each.replace("optim.", "").replace("torch.", "")
            performance_data[nice_label] = results[i]

        # ########################
        # Learning rate variation
        # #########################
        rosen_list = ['1000.0', '100.0', '10.0', '1.0']
        plot_data = {}
        for n_plot, b in enumerate(rosen_list):
            for each in algorithm_list:
                nice_label = each.replace("optim.", "").replace("torch.", "")
                file_name = f'data/{nice_label}_{b}_hypertrack.json' 
                with open(file_name, "r") as json_file:
                    data = json.load(json_file)
                    if n_plot == 0:
                        plot_data[nice_label] = [data["Best"]["lr"]]
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
        
        # #################
        # Chart generation
        # #################

        # Convert dictionaries into lists
        keys = list(performance_data.keys())
        values1 = list(performance_data.values())
        values2 = list(relative_dict.values())

        # Create scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(values1, values2, color='blue', label='Data Points')

        # Annotate points with keys as labels
        for i, key in enumerate(keys):
            plt.text(values1[i], values2[i], str(key), fontsize=10, ha='right')

        # Add labels and legend
        plt.xlabel('Values from dict1')
        plt.ylabel('Values from dict2')
        plt.title('Scatter Plot with Keys as Labels')
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True)

        # Adjust layout and show
        plt.tight_layout()
        plt.savefig("img/Smart_plot_3.pdf", format="pdf", dpi=300)
        plt.close()

    
    if False:
        fig, ax = plt.subplots(2, 3, figsize=(12, 6), height_ratios=(4, 2))

        plot_portrait("AdaBound_1.0_optimtrack", ax[0, 0])
        plot_portrait("AdaBound_100.0_optimtrack", ax[0, 1])
        plot_portrait("AdaBound_1000.0_optimtrack", ax[0, 2])
        plot_loss("AdaBound_1.0_optimtrack", ax[1, 0])
        plot_loss("AdaBound_100.0_optimtrack", ax[1, 1])
        plot_loss("AdaBound_1000.0_optimtrack", ax[1, 2])

        plt.tight_layout(pad=0.1)
        plt.savefig("img/AdaBound.pdf", format="pdf", dpi=300)
        plt.close()

    if False:
        for each in algorithm_list:
            nice_label = each.replace("optim.", "").replace("torch.", "")
            fig, ax = plt.subplots(figsize=(12, 6))
            plot_loss(f'{nice_label}_100.0_optimtrack', ax, n_trajectories=1000)
            plt.tight_layout(pad=0.1)
            plt.savefig(f'img/1_{nice_label}.pdf', format="pdf", dpi=300)
            plt.close()
    