[![DOI](https://zenodo.org/badge/656170700.svg)](https://doi.org/10.5281/zenodo.14279352)
# Experiments of optimization algorithms performance
The repository content the code of experiments to compare optimization algorithms performance.<br />
The reference article will be available soon. 
# How to launch the experiments
1. Be sure to have Python 3.9 and install dependences with Poetry package manager:
```
poetry install
```

2. Run the experiments:
```
poetry run python benchmark_clear.py
poetry run python benchmark_final.py
poetry run python benchmark_resnet.py
```
3. The result are saved in corresponded folders: data, data_without, data_ResNet.
4. Plot the fetched data:
```
poetry run python benchmark_smart_plot.py
```
Charts demonstrated in the paper will be saved in img folder.

Thank you for your attention!
