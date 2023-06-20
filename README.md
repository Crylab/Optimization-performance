# Experiments of optimization algorithms performance
The repository content the code of experiments to compare optimization algorithms performance.<br />
The reference article will be available soon. 
# How to launch the experiments[1 way: on your machine]
1. Be sure to have Python 3.9 and install dependences:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip3 install torch_optimizer
pip3 install hyperopt
pip3 install scikit-learn
```

2. Run the experiments:
```
python benchmark.py
```
3. Fetch the results:
At the end of the experiment, the set of .txt files is generated.
- *Rosenbrock function with hyper-optimizationarg_list.txt*<br />
  content the argument of performance curves with hyperparameter optimization
- *Rosenbrock function with hyper-optimizationlabel_list.txt*<br />
  content the labels of performance curves with hyperparameter optimization
- *Rosenbrock function with hyper-optimizationvalues_list.txt*<br />
  content the values of performance curves with hyperparameter optimization
- *Slope.txt*<br />
  content the linearized inclanation of the performance curve with hyperparameter optimization
- *Rosenbrock function without hyper-optimizationarg_list.txt*<br />
  content the argument of performance curves without hyperparameter optimization
- *Rosenbrock function without hyper-optimizationlabel_list.txt*<br />
  content the labels of performance curves without hyperparameter optimization
- *Rosenbrock function without hyper-optimizationvalues_list.txt*<br />
  content the values of performance curves without hyperparameter optimization
- *Slope_robust.txt*<br />
  content the linearized inclanation of the performance curve without hyperparameter optimization
- *ParamPath.txt*<br />
  content the hyperparameters for all argument values for all algorithms
4. Plot the fetched data:
```
with open('Rosenbrock function without hyper-optimizationarg_list.txt', 'r') as f:
  arr_x = json.load(f)
with open('Rosenbrock function without hyper-optimizationlabel_list.txt', 'r') as f:
  label = json.load(f)
with open('Rosenbrock function without hyper-optimizationvalue_list.txt', 'r') as f:
  arr = json.load(f) 
fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(len(label)):
  lab = label[i].split(".")[-1]
  ax.plot(arr_x, arr[i], label=lab)
plt.show()
```
Graphs of other data are plotted in the same way.

# How to launch the experiments[2 way: Dockerfile]
1. Build the Docker image
```
docker build -t image_name . --file Dockerfile
```
2. Run the Docker image
```
docker run -d --name my-container image_name
```
3. Fetch and plot
The data will be generated in the container. It is possible to collect it and plot in way described above.

# How to launch the experiments[3 way: Precomputed]
1. Download the data<br />
   The data was precomputed in transparent way by means of GitHub Actions. The data available in Latest Release in Zip archaive.
3. Unzip arhavive and plot<br />
   It is possible to unzip archaive and plot in way described above.
