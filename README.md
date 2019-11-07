# Deployment of Field Experiments in the Continuous Space  

Reference material for the submission of the manuscript
The mLG-HOO algorithm together with different simulation cases for multi-dimensional optimization.

## Dependencies:
* Python 3.6
* Numpy
* Nelder-Mead - from scipy
* Tree Parzen estimator - https://github.com/Dreem-Organization/benderopt/
* Jupyter Notebook
* R with the packages from the script

## Note
The code that runs the comparison is available in the Jupyter notebook. The saved data is inside the folder and the statistical analysis is available in the R file.

Each run might take some time to compute all cases. The whole notebook took ~48h to run on a 8GB Macbook. Most of the time is due to the TPE algorithm as the number of dimensions increase.