# Identifying Recurring and Missed Payments in Bank Statements: A Graph-Based Approach
Author: Jason Young


This folder contains a research paper, code, and support files for the author's practicum for the Georgia Tech Master of Science in Computational Analytics.

To learn more about the methodology, please read the paper,  'Recurring and Missed Payment Detection Paper.pdf' file. 

## Setup
To setup and run the code in an anaconda environment, follow the instructions in the /environment folder.

## Organizaton
Code for this project is saved in 2 files:
* JY_Utils.py  - contains the 2 custom ML algorithms for identifying recurring and late payments, plus a variety utility functions 
* Experiments.ipynb - is a jupyternotebook file demonstrating the methodology and creating visuals used in the paper

The code occasionaly reads from and writes to the /private folder, which is unpublished. This folder contains all confidential data. This includes the original dataset in .csv format, a list of sampled applicants in .csv (for reproducing results), and all intermediate Excel (.xlsx) files that were used to manually evaluate the results of the algorithms in Section 3.3.

 The /images folder contains visualization outputs of Experiments.ipynb that were used in the paper.

