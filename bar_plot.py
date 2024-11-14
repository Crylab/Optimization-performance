import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import json
from sklearn.linear_model import LinearRegression
from matplotlib.colors import LogNorm
import numpy as np
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
    fig.suptitle("Performance on Rosenbrock function with hyperparameters optimization")
    fig.subplots_adjust(wspace=0.005, hspace=0.1)
    plt.savefig(f"img/Chart_with_opt.pdf")
    print(f"Look at the picture: img/Chart_with_opt.pdf")

def three_bar():
    # Data
    with open("Rosenbrock function with hyper-optimizationarg_list.txt", 'r') as f:
        argument = json.load(f)
    with open("Rosenbrock function with hyper-optimizationlabel_list.txt", 'r') as f:
        label = json.load(f)
    with open("Rosenbrock function with hyper-optimizationvalue_list.txt", 'r') as f:
        value = json.load(f)

    categories = []
    values = []
    for i, each in enumerate(label):
        values.append(value[i][1])
        nice_label = each.replace("optim.", "")
        nice_label = nice_label.replace("torch.", "")
        categories.append(nice_label)
    #values = np.random.randint(5, 20, size=23)  # Generating random values for demonstration

    # Colors and aesthetics
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(values)))  # Using 'viridis' colormap for variety
    bar_width = 0.6

    # Plot
    list_of_indexes = [1, 4, 9]
    fig, ax = plt.subplots(3, 1, figsize=(8, 12))
    for i in range(3):
        values = []
        for j in range(len(categories)):
            values.append(value[j][list_of_indexes[i]])

        bars = ax[i].bar(categories, values, color=colors, width=bar_width)

        # Customizing each bar's appearance
        for bar in bars:
            bar.set_edgecolor('k')
            bar.set_linewidth(1)


        # Title and labels
        #ax.set_title('Performance on Rosenbrock function with hyperparameters optimization', fontsize=16, fontweight='bold')
        ax[i].set_ylabel(f'Loss value for b={argument[list_of_indexes[i]]:.4}', fontsize=14)
        ax[i].set_yscale("log")

        # Rotate category labels for readability
        for label in ax[i].get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment('right')
            label.set_fontsize(14)
        for label in ax[i].get_yticklabels():
            label.set_fontsize(12)

        ax[i].grid(True, linestyle='--', linewidth=0.7, color='gray', alpha=0.7)


        # Aesthetic adjustments
        #plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(f"img/BarChart_with_opt.pdf")
    print(f"Look at the picture: img/BarChart_with_opt.pdf")

def dummy_bar():
    # Data
    with open("Rosenbrock function with hyper-optimizationarg_list.txt", 'r') as f:
        argument = json.load(f)
    with open("Rosenbrock function with hyper-optimizationlabel_list.txt", 'r') as f:
        label = json.load(f)
    with open("Rosenbrock function with hyper-optimizationvalue_list.txt", 'r') as f:
        value = json.load(f)

    val_dict = {}

    
    for i, each in enumerate(label):
        nice_label = each.replace("optim.", "")
        nice_label = nice_label.replace("torch.", "")
        val_dict[nice_label] = value[i].copy()

    sorted_dict = dict(sorted(val_dict.items(), key=lambda item: item[1][-1]))

    # Colors and aesthetics
    bar_width = 0.6

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    for i in reversed(range(10)):
        categories = []
        values = []
        for key, value in sorted_dict.items():
            categories.append(key)
            values.append(value[i])
            bars = ax.bar(categories, values, color=plt.cm.viridis(float(i/10.0)), width=bar_width)        

        # Customizing each bar's appearance
        for bar in bars:
            bar.set_edgecolor('k')
            bar.set_linewidth(1)


    # Title and labels
    #ax.set_title('Performance on Rosenbrock function with hyperparameters optimization', fontsize=16, fontweight='bold')
    ax.set_ylabel('Loss value', fontsize=14)
    ax.set_yscale("log")

    # Rotate category labels for readability
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment('right')
        label.set_fontsize(14)
    for label in ax.get_yticklabels():
        label.set_fontsize(12)

    ax.grid(True, linestyle='--', linewidth=0.7, color='gray', alpha=0.7)


        # Aesthetic adjustments
        #plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    custom_line = Line2D([0], [0], color=plt.cm.viridis(0.5), linewidth=3)

    # Add custom legend to the plot
    red_patch = mpatches.Patch(color='blue', label='Fragmented', alpha=0.3)
    blue_patch = mpatches.Patch(color='orange', label='Segmented', alpha=0.3)

    ax[1].legend(handles=[red_patch, blue_patch])
    ax.legend([custom_line], ['Label'], loc='upper right', fontsize=12)


    plt.savefig(f"img/Dummy_bar.pdf")
    print(f"Look at the picture: img/Dummy_bar.pdf")


    # Define the Rosenbrock function
def rosenbrock(x, y, b=100):
    return (1 - x)**2 + b * (y - x**2)**2

def rosen_visualization():

    # Generate grid data for x and y
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-2, 2, 400)
    X, Y = np.meshgrid(x, y)

    # Set custom boundaries for the colormap
    vmin = 0.1  # Avoid log(0) by setting a minimum value greater than 0
    vmax = 2000

    # Create the figure and subplots (2 plots in one figure)
    fig, axes = plt.subplots(1, 3, subplot_kw={'projection': '3d'}, figsize=(15, 5))

    b_list = [1, 10, 100]

    for i in range(3):
        # Calculate Z values using the Rosenbrock function
        Z = rosenbrock(X, Y, b_list[i])
        vmin = 0.1
        vmax = np.max(Z)

        norm = LogNorm(vmin=vmin, vmax=vmax)

        # Plot the first surface on the first axis
        surf = axes[i].plot_surface(X, Y, Z, cmap='viridis', norm=norm)

        axes[i].set_title(f'Rosenbrock function with b={b_list[i]} \n $f(x, y) = (1 - x)^2 + {b_list[i]}(y - x^2)^2$')
        axes[i].set_xlabel('X')
        axes[i].set_ylabel('Y')
        axes[i].set_zlabel('f(x, y)')
        axes[i].view_init(elev=30, azim=60)  # Adjust the view angle for this subplot
        fig.colorbar(surf, ax=axes[i])

    # Display the plot
    plt.tight_layout()
    plt.savefig(f"img/rosen.pdf")
    print(f"Look at the picture: img/rosen.pdf")


if __name__ == "__main__":
    #three_bar()
    #dummy_bar()
    rosen_visualization()
    
    
    
    