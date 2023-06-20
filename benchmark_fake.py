file_list = ["ParamPath.txt", "Slope.txt", "Rosenbrock function with hyper-optimizationarg_list.txt", "Rosenbrock function with hyper-optimizationlabel_list.txt", "Rosenbrock function with hyper-optimizationvalue_list.txt", "Rosenbrock function without hyper-optimizationarg_list.txt", "Rosenbrock function without hyper-optimizationlabel_list.txt", "Rosenbrock function without hyper-optimizationvalue_list.txt"]

if __name__ == "__main__":
    print("Hello World!")
    for name in file_list:
        with open(name, "w") as f:
          f.write("Very reasonable data!")
