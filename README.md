# ML4FF

This repository contains the framework code and dataset used in the paper “ML4FF: A machine-learning framework for flash flood forecasting applied to a Brazilian watershed” by Jaqueline A. J. P. Soares, Luan C. S. M. Ozelim, Luiz Bacelar, Dimas B. Ribeiro, Stephan Stephany, and Leonardo B. L. Santos.

# Project tree

 * [code](/../../tree/main/code)
   * [ML4FF.py](/../../blob/main/code/ML4FF.py)
   * [requirements.txt](/../../blob/main/code/requirements.txt)
   * [environment.yml](/../../blob/main/code/environment.yml)
 * [data](/../../tree/main/data)
   * [data.csv](/../../blob/main/data/data.csv)
   * [data_columns.csv](/../../blob/main/data/data_columns.csv)
 * [figs](/../../tree/main/figs)
 * [results](/../../tree/main/results)
   * [Models](/../../tree/main/results/Models)
   * [Summary.xlsx](/../../blob/main/results/Summary.xlsx)
   * [Summary_Perf.xlsx](/../../blob/main/results/Summary_Perf.xlsx)
 * [LICENSE](/../../blob/main/LICENSE)
 * [README.md](/../../blob/main/README.md)

**code**

The code folder contains the code and requirements needed to execute the ML4FF framework. 

The ML4FF.py file contains the source Python code of the ML4FF framework. The code is self-explanatory, allowing users to input their own datasets and train and optimize a diverse set of 34 machine learning (ML) models across 11 different classes, choosing the best-performing models.

The requirements.txt file lists the necessary Python packages required to set up an environment for running the framework. Users can use this file to install the dependencies and run the framework as shown below:

# python "ML4FF.py" --dataset data/data.csv --columns data/data_columns.csv --output "D:\\ML4FF" --save_models --inner_cv 10 --outer_cv 30 --holdout_slice 0.875 --seed_ml 10 --seed_dl 0

For Conda users, the environment.yml file specifies the environment configuration, including Python version 3.10.12. This file can be used to create the environment with the following command:

# conda env create -f environment.yml
# conda activate ml4ff_env

**data**

The data folder contains the dataset related to the test case for the Brazilian watershed in CSV format (data.csv).

The data_columns.csv file is used as input in the command line to run the framework. It can be customized to suit the user's specific requirements. This file contains just the columns that should be used from the dataset file. The first column is the index, and the last column is used as output.

**figs**

The figs folder contains PDF-format files of the figures generated in the research for the presented test case.

**results**

Summary.xlsx and Summary_Perf.xlsx are two Excel spreadsheets created using the functions build_excel and perf_excel, respectively, of the ML4FF.py file. The use of such functions is documented within the ML4FF.py file. In short, these spreadsheets summarize metrics and statistics on the application of the framework to the test case for the Brazilian watershed. 

The Models folder contains the pickled results obtained by the 34 ML methods available in the framework applied to the test case. These files can be imported and manipulated, as documented within the Python framework code.
