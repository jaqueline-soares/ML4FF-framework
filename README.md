# ML4FF

This repository contains all the framework codes and the dataset used in the paper “ML4FF: A machine-learning framework for flash flood forecasting applied to a Brazilian watershed” by Jaqueline A. J. P. Soares, Luan C. S. M. Ozelim, Luiz Bacelar, Dimas B. Ribeiro, Stephan Stephany, and Leonardo B. L. Santos.

# Project tree

 * [code](/../../tree/main/code)
   * [ML4FF.py](/../../blob/main/code/ML4FF.py)
   * [environment.yml](/../../blob/main/code/environment.yml)
   * [requirements.txt](/../../blob/main/code/requirements.txt)
   * [config.json](/../../blob/main/code/config.json)
 * [data](/../../tree/main/data)
   * [data.csv](/../../blob/main/data/data.csv)
 * [figs](/../../tree/main/figs)
 * [results](/../../tree/main/results)
   * [Models](/../../tree/main/results/Models)
   * [Summary.xlsx](/../../blob/main/results/Summary.xlsx)
   * [Summary_Perf.xlsx](/../../blob/main/results/Summary_Perf.xlsx)
 * [LICENSE](/../../blob/main/LICENSE)
 * [README.md](/../../blob/main/README.md)

## code

The code folder contains the source code, a sample configuration file, and the required dependencies to run the ML4FF framework.

The ML4FF.py file contains the source Python code of the ML4FF framework. The code is self-explanatory, allowing users to input their own datasets and train and optimize a diverse set of 34 machine learning (ML) models across 11 different classes, choosing the best-performing models.

To run the framework, a **JSON configuration file** is required. This file defines the parameters necessary for the experiment and allows users to customize settings based on their specific needs. The provided config.json file represents the configuration used in the paper. The following table explains each parameter in the configuration file:

| Parameter                    | Description |
|------------------------------|-------------|
| `dataset_path`               | Path to the CSV file containing the data. |
| `dataset_columns`            | List with just the columns that should be used from the dataset. **Important:** the <u>first column</u> is the index, and the <u>last column</u> is used as output. |
| `result_path`                | Path to store the output files. |
| `save_models`                | Indicates whether trained models should be saved (`true` or `false`). |
| `inner_cv`                   | Number of splits for inner cross-validation. |
| `outer_cv`                   | Number of splits for outer cross-validation. |
| `holdout_slice`              | Percentage of data used as holdout. |
| `seed`                       | Seed to ensure experiment reproducibility. |
| `ml_algorithms`              | List of Machine Learning algorithms to be used. |
| `dl_algorithms`              | List of Deep Learning algorithms to be used. |

Users can run the framework using the following command:

```bash
python ML4FF.py -c config.json
```

The `requirements.txt` file lists all the necessary Python packages to set up the environment for running the framework. Users can install these dependencies with the following:

```bash
pip install -r requirements.txt
```

For Conda users, the `environment.yml` file provides the environment configuration, including Python version 3.10.12. To create and activate the environment, run:

```bash
conda env create -f environment.yml
conda activate ml4ff_env
```

## data

The data folder contains the dataset related to the test case for the Brazilian watershed in CSV format (data.csv).

Users are encouraged to contribute to this framework by suggesting new datasets to be uploaded here.

## figs

The figs folder contains PDF-format files of the figures generated in the research for the presented test case.

## results

Summary.xlsx and Summary_Perf.xlsx are two Excel spreadsheets created using the functions build_excel and perf_excel, respectively, of the ML4FF.py file. The use of such functions is documented within the ML4FF.py file. In short, these spreadsheets summarize metrics and statistics on the application of the framework to the test case for the Brazilian watershed. 

The Models folder contains the pickled results obtained by the 34 ML methods available in the framework applied to the test case. These files can be imported and manipulated, as documented within the Python framework code.
