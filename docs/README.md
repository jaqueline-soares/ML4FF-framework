# ML4FF

This repository contains the framework code and dataset used in the paper “ML4FF: A machine-learning framework for flash flood forecasting applied to a Brazilian watershed” by Jaqueline A. J. P. Soares, Luan C. S. M. Ozelim, Luiz Bacelar, Dimas B. Ribeiro, Stephan Stephany and Leonardo B. L. Santos.

# Project tree

 * [ML4FF.py](/../../blob/main/ML4FF.py)
 * [Summary.xlsx](/../../blob/main/Summary.xlsx)
 * [Summary_Perf.xlsx](/../../blob/main/Summary_Perf.xlsx)
 * [data](/../../tree/main/data)
 * [docs](/../../tree/main/docs)
 * [figs](/../../tree/main/figs)
 * [Models](/../../tree/main/Models)
 * [LICENSE](/../../blob/main/LICENSE)
 * [README.md](/../../blob/main/docs/README.md)

The ML4FF.py file contains the source Python code of the ML4FF framework. The code is self-explanatory, allowing researchers to reproduce the results presented in the corresponding paper.

Summary.xlsx and Summary_Perf.xlsx are two Excel spreadsheets created using the functions build_excel and perf_excel, respectively, of the ML4FF.py file. The use of such functions is documented within the ML4FF.py file. In short, these spreadsheets summarize metrics and statistics on the application of the framework to the test case for the Brazilian watershed. 

The data folder contains the dataset related to the test case for the Brazilian watershed in CSV format.

The docs folder contains the README file with general instructions.

The figs folder contains PDF-format files of all the figures generated in the research, including those not presented in the paper due to space limitations.

The Models folder contains the pickled results obtained by the 32 ML methods available in the framework applied to the test case. These files can be imported and manipulated, as documented within the Python framework code.
