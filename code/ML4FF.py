# Import all packages needed

from datetime import timedelta
import numpy as np
import os
import pandas as pd
import pickle
import psutil
from hpsklearn import all_regressors
import hyperopt
from hyperopt import fmin,tpe,space_eval,hp
from hyperopt.early_stop import no_progress_loss
from scipy.stats import bootstrap
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from tempfile import mkdtemp
import time
import shutil
from joblib import dump, load
import torch
from torch import nn
import torch.nn.functional as F
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping
import argparse
import json
from dataclasses import dataclass, field
from typing import List
import random

@dataclass
class Config:
    dataset_path: str
    dataset_columns: List[str]
    result_path: str
    save_models: bool
    inner_cv: int
    outer_cv: int
    holdout_slice: float
    seed: int
    ml_algorithms: List[str]
    dl_algorithms: List[str]

    @staticmethod
    def from_json(file_path: str) -> "Config":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Config(**data)
    
class DatasetHandler:

    def __init__(self, config):
        # Load the dataset from the path defined in the config file.
        self.ML4FF_dataset = pd.read_csv(config.dataset_path)

        if not set(config.dataset_columns).issubset(self.ML4FF_dataset.columns):
            raise ValueError(f"Some specified columns are not in the dataset. Missing columns: {set(config.dataset_columns) - set(self.ML4FF_dataset.columns)}")

        # Filter the dataset columns with the informed columns
        self.ML4FF_dataset = self.ML4FF_dataset[config.dataset_columns]

        # Set the first specified column as the index of the dataframe
        self.ML4FF_dataset = self.ML4FF_dataset.set_index(config.dataset_columns[0])

        # Input size is the number of columns without the output
        self.input_size = len(self.ML4FF_dataset.columns) - 1

        # Split the features and the outputs to be predicted.
        features = self.ML4FF_dataset[config.dataset_columns[1:-1]].to_numpy()
        outputs = self.ML4FF_dataset[config.dataset_columns[-1]].to_numpy()

        self.features_nestedCV32 = features[:int(len(features)*config.holdout_slice)].astype(np.float32)
        self.outputs_nestedCV32 = outputs[:int(len(features)*config.holdout_slice)].astype(np.float32)
        self.features_holdout32 = features[int(len(features)*config.holdout_slice):].astype(np.float32)
        self.outputs_holdout32 = outputs[int(len(features)*config.holdout_slice):].astype(np.float32)

        self.features_nestedCV64 = features[:int(len(features)*config.holdout_slice)]
        self.outputs_nestedCV64 = outputs[:int(len(features)*config.holdout_slice)]
        self.features_holdout64 = features[int(len(features)*config.holdout_slice):]
        self.outputs_holdout64 = outputs[int(len(features)*config.holdout_slice):]

    def get_data(self, mdl_name):
        if mdl_name[:2]=="DL":
            return self.features_nestedCV32, self.outputs_nestedCV32, self.features_holdout32, self.outputs_holdout32
        else:
            # Using the data as float64 is important for the reproducibility of the ML algorithms
            return self.features_nestedCV64, self.outputs_nestedCV64, self.features_holdout64, self.outputs_holdout64
        
    def get_input_size(self):
        return self.input_size
    
    def get_dataset(self):
        return self.ML4FF_dataset

# Define auxiliary functions to be used in the routine:

# CI_Scipy calculates the 95% confidence BCa bootstrap confidence interval of the mean of the input values "vals"
def CI_Scipy(vals):
    vals = (vals,)
    res = bootstrap(vals, np.mean, n_resamples=1000,confidence_level=0.95,random_state=123).confidence_interval
    return [res[0],np.mean(vals),res[1]]

# nse calculates the Nash-Sutcliffe efficiency coefficient for the predicted values "predictions" relative to the "targets".
def nse(predictions, targets):
    return (1-(np.sum((targets-predictions)**2)/np.sum((targets-np.mean(targets))**2)))

# rmse calculates the RMSE between the predicted values "a1" and the observed ones "a2".
def rmse(a1,a2):
    return (np.mean((a1-a2)**2))**0.5

# min_max_med_var calculates the minimum, maximum, median and variance of dataset "x".
def min_max_med_var(x):
    return [np.min(x),np.max(x),np.median(x),np.var(x)]

# kge calculates the Kling-Gupta efficiency coefficient for the predicted values "simulations" relative to the "evaluation".
def kge(simulations, evaluation):
    """Implementation taken from Hallouin (2021) - https://doi.org/10.5281/zenodo.2591217
    """
    # calculate error in timing and dynamics r
    # (Pearson's correlation coefficient)
    sim_mean = np.mean(simulations, axis=0, dtype=np.float64)
    obs_mean = np.mean(evaluation, dtype=np.float64)

    r_num = np.sum((simulations - sim_mean) * (evaluation - obs_mean),
                   axis=0, dtype=np.float64)
    r_den = np.sqrt(np.sum((simulations - sim_mean) ** 2,
                           axis=0, dtype=np.float64)
                    * np.sum((evaluation - obs_mean) ** 2,
                             dtype=np.float64))+10**(-10)
    r = r_num / r_den
    # calculate error in spread of flow alpha
    alpha = np.std(simulations, axis=0) / np.std(evaluation, dtype=np.float64)
    # calculate error in volume beta (bias of mean discharge)
    beta = (np.sum(simulations, axis=0, dtype=np.float64)
            / np.sum(evaluation, dtype=np.float64))
    # calculate the Kling-Gupta Efficiency KGE
    kge_ = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    return kge_

# Build the list "all_algs" with all the regressors in hpsklearn, specially their names, class and dictionaries with its input parameters and Bayesian search ranges:

def buil_dict(mdl_base):
    lst = mdl_base.named_args
    dictn={}
    for ll in lst:
        dictn[ll[0]]=ll[1]
    return dictn

class LSTMModel(nn.Module):
    def __init__(self, 
                 input_size, 
                 hidden_size = 10, 
                 num_layers = 3, 
                 dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

class LSTMPreModel(nn.Module):
    def __init__(self, 
                 seq_len = 10,
                 hidden_size = 12, 
                 num_layers = 3, 
                 dropout=0.2,
                 num_units = 10,
                 nonlin = F.relu):
        super(LSTMPreModel, self).__init__()
        self.dense0 = nn.Linear(seq_len, num_units)
        self.nonlin = nonlin
        self.dense1 = nn.Linear(num_units, seq_len)
        
        self.lstm = nn.LSTM(seq_len, hidden_size = hidden_size, num_layers = num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size*seq_len, 1)

    def forward(self, x):
        bs,sl = x.shape
        x = x.unsqueeze(1).repeat(1, sl, 1)
        x = self.nonlin(self.dense0(x))
        x = self.nonlin(self.dense1(x))
        x, _ = self.lstm(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x    

# We need to define the InOrderSplit class to activate an EarlyStopping callback which respects order of the data (in this case, val_frac % of the data for validation loss monitoring).
    
class InOrderSplit:
    def __init__(self, val_frac):
        self.val_frac = val_frac

    def __call__(self, dataset,y):
        len_dataset = len(dataset)
        len_val = int(len_dataset * self.val_frac)
        len_train = len_dataset - len_val
        train_idx = list(range(0, len_train))
        val_idx = list(range(len_train, len_dataset))
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)
        return train_dataset, val_dataset
    
# The function RefineNCV_General generates two outputs, namely: resultados and resultados_stack.
# "resultados": is a list containing, in this order:
#  a) The model name
#  b) Identification of the dataset considered (either "CV_outer-X" where X is the outer folder number or "Holdout-" if holdout set)
#  c) Numpy array containing the differences between predictions and observed values for the dataset considered
#  d) Best hyperparameters found in the loop for the dataset considered
#  e) Mean Absolute Error between the predictions and observed values for the dataset considered

# "resultados_stack": is a list containing, in this order:
#  a) The model name
#  b) Identification of the dataset considered (either "CV_outer-X" where X is the outer folder number or "Holdout-" if holdout set)
#  c) Numpy array containing two sub-arrays: the predictions and the observed values for the dataset considered
#  d) Best hyperparameters found in the loop for the dataset considered (string version for DL)
#  e) ETTrain (time in seconds to perform the Nested CV + train model on full Nested-CV dataset)
#  f) ETHoldout (time in seconds to perform the predictions on the holdout dataset)
#  *g) VMS memory difference in MB from the beginning to the end of the Nested CV + train model on full Nested-CV dataset prodecure
#  *h) RSS memory difference in MB from the beginning to the end of the Nested CV + train model on full Nested-CV dataset prodecure
#  *i) VMS memory difference in MB from the beginning to the end of the prediction process on the holdout dataset
#  *j) RSS memory difference in MB from the beginning to the end of the prediction process on the holdout dataset
# *items from g) to j) are not reliable, unless you running the code on a dedicated virtual machine.

# The inputs of the RefineNCV_General function are:
# a) "datasetHandler": dataset handler to obtain the features and output values (values to be predicted) 
# b) "random_state": random state to make the results reproducible
# c) "inner_folds": number of inner folds in the Nested-CV (in the case of the paper, 10)
# d) "outer_folds": number of outer folds in the Nested-CV (in the case of the paper, 30)
# e) "all_candidates": list of algorithims to be considered in the benchmark.
# f) "root": path where the pickled values of "resultados" and "resultados_stack" will be stored for each method. 
# g) "save_m": flag that indicates if the production pipeline will be saved to root+"\\Models_Production\\ after creation. 

# We need this auxiliary function to properly export and save the string versions of hyperparameters of the DL network.
def filt_dict_DL(dctpars):
    keys_v = dctpars.keys()
    dctpars_out = {}
    for keyy in keys_v:
        dctpars_out[keyy]=str(dctpars[keyy])
    return dctpars_out

def RefineNCV_General(datasetHandler,random_state,inner_folds,outer_folds,all_candidates,root,save_m):   
    try:
        os.mkdir(root)
    except:
        pass
    
    try:
        os.mkdir(os.path.join(root,"Models"))
    except:
        pass

    try:
        os.mkdir(os.path.join(root,"Temp"))
    except:
        pass
    
    resultados=[]
    resultados_stack=[]
     
    cv_inner = TimeSeriesSplit(n_splits=inner_folds)
    cv_outer = TimeSeriesSplit(n_splits=outer_folds)   
    process = psutil.Process(os.getpid())
        
    def my_custom_loss_func(targets_s, predictions_s):
        nse = (1-(np.sum((targets_s-predictions_s)**2)/np.sum((targets_s-np.mean(targets_s))**2)))
        return 1/(2-nse)
    
    for mdl_info in all_candidates:

        reset_seed(random_state)

        mdl_name,mdl_type,space = mdl_info
        print(mdl_name)

        features_nestedCV, outputs_nestedCV, features_holdout, outputs_holdout = datasetHandler.get_data(mdl_name)
        X_A = features_nestedCV
        y_A = outputs_nestedCV

        try:
            def inner_Search(X_Avi,y_Avi):
                if mdl_name[:2]=="DL":
                    mdl = mdl_type
                    X_Av = X_Avi.astype(np.float32)
                    y_Av = y_Avi.reshape(-1,1).astype(np.float32)
                else:
                    mdl = hyperopt.pyll.stochastic.sample(mdl_type)
                    X_Av = X_Avi
                    y_Av = y_Avi
                def objective(params,X_train_i=X_Av,y_train_i=y_Av,tscv=cv_inner):
                    cachedir = mkdtemp(dir=os.path.join(root,"temp"))
                    pipeline = Pipeline([('transformer', MinMaxScaler()), ('estimator', mdl.set_params(**params))],memory=cachedir)
                    scoring = make_scorer(my_custom_loss_func, greater_is_better=True)
                    try:
                        scr = -cross_val_score(pipeline, X_train_i, y_train_i, cv = tscv,n_jobs=-1,scoring=scoring).mean()
                    finally:
                        shutil.rmtree(cachedir)  # Remove the temporary directory after use
                    return scr

                ########################################################################################
                ################## Insert your customizations in the ML methods here! ##################
                ########################################################################################

                best=fmin(fn=objective, 
                        space=space, 
                        algo=tpe.suggest, 
                        max_evals=100, 
                        early_stop_fn=no_progress_loss(10),
                        rstate=np.random.default_rng(random_state)
                      )
                best_par = space_eval(space, best)
                return mdl.set_params(**best_par)    

            pvi = 0
            start_time_outer = time.monotonic()
            start_vms_outer = process.memory_info().vms/(1024*1024)
            start_rss_outer = process.memory_info().rss/(1024*1024)
            for train_ix, test_ix in cv_outer.split(X_A):
                start_time = time.monotonic()
                start_vms = process.memory_info().vms/(1024*1024)
                start_rss = process.memory_info().rss/(1024*1024)
                stackinp=[]
                resulp=[]

                X_train, X_test = X_A[train_ix, :], X_A[test_ix, :]
                y_train, y_test = y_A[train_ix], y_A[test_ix]
                result = inner_Search(X_train, y_train)

                sc_X_A = MinMaxScaler()

                X_train_s = sc_X_A.fit_transform(X_train)
                X_test_s = sc_X_A.transform(X_test)
                if mdl_name[:2]=="DL":
                    result.fit(X_train_s, y_train.reshape(-1,1))
                else:
                    result.fit(X_train_s, y_train.ravel())
                end_time = time.monotonic()
                end_vms = process.memory_info().vms/(1024*1024)
                end_rss = process.memory_info().rss/(1024*1024)
                yhat = result.predict(X_test_s)
                end_time_2 = time.monotonic()
                end_vms_2 = process.memory_info().vms/(1024*1024)
                end_rss_2 = process.memory_info().rss/(1024*1024)
                stackinp.append([yhat.flatten(),y_test.flatten()])
                real_vals_diff = yhat.flatten()-y_test.flatten()
                resulp.append(real_vals_diff)

                if mdl_name[:2]=="DL":
                    parout = filt_dict_DL(result.get_params())
                else:
                    parout = result.get_params()

                resultados.append([mdl_name,"CV_outer-"+str(pvi),np.array(resulp),result.get_params(),np.mean(np.abs(real_vals_diff))])
                resultados_stack.append([mdl_name,"CV_outer-"+str(pvi),np.array(stackinp),parout,timedelta(seconds=end_time - start_time),
                                           timedelta(seconds=end_time_2 - end_time),end_vms-start_vms,end_rss-start_rss,
                                         end_vms_2-end_vms,end_rss_2-end_rss])
                pvi=pvi+1

            bst_mdl_hyp = inner_Search(X_A,y_A)

            sc_X_A = MinMaxScaler()

            X_A_s = sc_X_A.fit_transform(X_A)

            if mdl_name[:2]=="DL":
                bst_mdl = bst_mdl_hyp.fit(X_A_s,y_A.reshape(-1,1))
            else:
                bst_mdl = bst_mdl_hyp.fit(X_A_s,y_A.ravel())

            stackinp=[]
            resulp=[]
            end_time_outer = time.monotonic()
            end_vms_outer = process.memory_info().vms/(1024*1024)
            end_rss_outer = process.memory_info().rss/(1024*1024)
            self_evals = bst_mdl.predict(sc_X_A.transform(features_holdout))
            end_time_outer_2 = time.monotonic()
            end_vms_outer_2 = process.memory_info().vms/(1024*1024)
            end_rss_outer_2 = process.memory_info().rss/(1024*1024)
            stackinp.append([self_evals.flatten(),outputs_holdout])
            real_vals_diff = self_evals.flatten()-outputs_holdout
            resulp.append(real_vals_diff)

            if mdl_name[:2]=="DL":
                parout_o = filt_dict_DL(bst_mdl.get_params())
            else:
                parout_o = bst_mdl.get_params()

            resultados.append([mdl_name,"HoldOut-",np.array(resulp),bst_mdl.get_params(),np.mean(np.abs(real_vals_diff))])
            resultados_stack.append([mdl_name,"HoldOut-",np.array(stackinp),parout_o,timedelta(seconds=end_time_outer - start_time_outer),
                                     timedelta(seconds=end_time_outer_2 - end_time_outer),end_vms_outer-start_vms_outer,end_rss_outer-start_rss_outer,
                                     end_vms_outer_2-end_vms_outer,end_rss_outer_2-end_rss_outer])   

            with open(root+"\\Models\\"+mdl_name+"-r", "wb") as f:
                pickle.dump([x for x in resultados if x[0]==mdl_name], f)
            with open(root+"\\Models\\"+mdl_name+"-rs", "wb") as f:
                pickle.dump([x for x in resultados_stack if x[0]==mdl_name], f)
                
            pipeline_prod = Pipeline([('transformer',  sc_X_A), ('estimator', bst_mdl)])
            
            if save_m:    
                try:
                    os.mkdir(os.path.join(root,"Models_Production"))
                except:
                    pass 
                dump(pipeline_prod, root+"\\Models_Production\\"+str(mdl_name)+'_production.joblib')
                        
        except:
            pass
    return [resultados,resultados_stack]


# If the models have already been run, their pickled lists would be available. The pickled lists of the methods considered in the ML4FF paper are also in the github repo.
# The following auxiliary function is needed to gather the pickled data located in "path_v". Its inputs are:
# a) "all_candidates": list of all the models run
# b) "path_v": path of the pickled lists.

def buil_results_pkl(all_candidates,path_v):
    resu=[]
    resuls=[]
    for mdl_info in all_candidates:
        mdl_name,mdl_type,space = mdl_info
        try:
            with open(path_v+"\\"+mdl_name+"-r", "rb") as f:
                rpar = pickle.load(f)
            with open(path_v+"\\"+mdl_name+"-rs", "rb") as f:
                rspar = pickle.load(f)
            for rr1 in rpar:
                resu.append(rr1)
            for rr2 in rspar:
                resuls.append(rr2)
        except Exception as ex:             
            raise ValueError(f"Fail when loading the model '{mdl_name}'") from ex
    return [resu,resuls]

# If one wants to get access to production models and use them for novel predictions, if the flag "save_m" was set to True in RefineNCV_General, the function load_production_pipeline allows for a direct import of the production pipeline:
# The arguments of the load_production_pipeline function are:
# a) "path_v": path of the pickled pipelines (by default this is root+"\\Models_Production\\").
# b) "mdl_name": name of the model whose pipeline one wants to import.
# The output of the load_production_pipeline function is a sklearn pipeline with fitted entities (scaler and model).
# For example, load_production_pipeline(root+"\\Models_Production\\",'sklearn_MLPRegressor') would retrieve the production pipeline of obtained for MLPRegressor.

def load_production_pipeline(path_v,mdl_name):
    pipeline_prod = load(path_v+str(mdl_name)+'_production.joblib')
    return pipeline_prod

# On the other hand, in cases where the flag "save_m" was set to False in RefineNCV_General and one still wants to get access to pipeline fitted with the best hyperparameters found by ML4FF (i.e., found during the nested-CV loop), instead of having to re-run all the ML4FF calculations, it is possible to directly use the dictionary of saved hyperparameters.
# This can be achieved by using the function build_production_pipeline, whose input arguments are:
# a) "features": input features considered in the benchmark
# b) "outputs": output values (values to be predicted) considered in the benchmark
# c) "hold_out": percentage of values out of the holdout (in the case of the paper, 0.875)
# d) "all_candidates": list of algorithims which were considered in the benchmark.
# e) "root": path where the pickled values of "resultados" and "resultados_stack" were stored (after running RefineNCV_General). 
# f) "mdl_name": name of the model whose pipeline one wants to create.
# g) "save_m": flag that indicates if the production pipeline will be saved to root+"\\Models_Production\\ after creation. 
# The output of the build_production_pipeline function is a sklearn pipeline with fitted entities (scaler and model).

def build_production_pipeline(features,outputs,hold_out,all_candidates,root,mdl_name,save_m):
    
    features_nestedCV = features[:int(len(features)*hold_out)].astype(np.float32)
    outputs_nestedCV = outputs[:int(len(features)*hold_out)].astype(np.float32)
    
    ML4FF_pickled = buil_results_pkl(all_candidates,root+"\\Models")

    par_mdl_framework = [x for x in ML4FF_pickled[0] if (x[0]==mdl_name)and(x[1][:2]!='CV')][0][3]
    _,mdl_type,_ = [x for x in all_candidates if x[0]==mdl_name][0]

    if mdl_name[:2]=="DL":
        mdl_mdl = mdl_type
        X_Av = features_nestedCV.astype(np.float32)
    else:
        mdl_mdl = hyperopt.pyll.stochastic.sample(mdl_type)
        X_Av = features_nestedCV

    pipeline_unfitted = Pipeline([('transformer',  MinMaxScaler()), ('estimator', mdl_mdl.set_params(**par_mdl_framework))])

    if mdl_name[:2]=="DL":
        pipeline_prod = pipeline_unfitted.fit(X_Av,outputs_nestedCV.reshape(-1,1))
    else:
        pipeline_prod = pipeline_unfitted.fit(X_Av,outputs_nestedCV.ravel())

    if save_m:    
        try:
            os.mkdir(os.path.join(root,"Models_Production"))
        except:
            pass 
        dump(pipeline_prod, root+"\\Models_Production\\"+str(mdl_name)+'_production.joblib')
    return pipeline_prod
    
# Two new auxiliary functions can be defined to process the results and save them in a Excel spreadsheet for better visualization of the results.
# The firs function is perf_excel, which builds a self-explanatory excel spreadsheet with performance metrics. Its inputs are:
# a) "simu_runs": the results from the simulations (either run_ML4FF or ML4FF_pickled).
# b) "root": root path where to save the Excel spreadsheet.

def perf_excel(simu_runs,root):
    lst_perf=[]
    lst_comp_perf=[]
    methds = list(set([x[0] for x in simu_runs[1]]))
    for mth in methds:
        rrr = [x for x in simu_runs[1] if x[0]==mth]
        dfperf = pd.DataFrame()
        for itv in rrr:
            algo,part,_,_,tcv,tpred,mvms_cv,mrss_cv,mvms_pred,mrss_pred = itv
            dfperf[part]=[tcv.total_seconds(),tpred.total_seconds(),mvms_cv,mrss_cv,mvms_pred,mrss_pred]
            dfperf.index = ["Loop time","Predict time","VMS_loop","RSS_loop","VMS_pred","RSS_pred"]
            lst_perf.append([mth.split("_")[-1],dfperf])
        dfn = pd.DataFrame([dfperf["HoldOut-"].to_numpy()])
        dfn.columns = dfperf.index
        dfn.index = [mth.split("_")[-1]]
        lst_comp_perf.append(dfn)
    with pd.ExcelWriter(root+'\\Summary_Perf.xlsx') as writer:
        final_eval = pd.concat(lst_comp_perf)
        final_eval.to_excel(writer, sheet_name='Compilation')
        for dfi in lst_perf:
            if dfi[0] in list(final_eval.index):
                dfi[1].to_excel(writer, sheet_name=dfi[0])
            else:
                pass
            
# Finally, the error metrics can be gathered and summarized in an Excel spreadsheet using the function build_excel. Its inputs are:
# a) "ML4FF_dataset": full dataset present in the github repo and used in the paper
# b) "ncvs": number of outer folds in the nested-CV (in the paper, 30)
# c) "hold_out": percentage of values out of the holdout (in the case of the paper, 0.875)
# d) "simu_runs": the results from the simulations (either run_ML4FF or ML4FF_pickled).

def build_excel(ML4FF_dataset,root,ncvs,hold_out,simu_runs):
    dfs=[]
    dfs2=[]
    dfs3=[]
    cols = ML4FF_dataset.index[int(len(ML4FF_dataset)*hold_out):]
    for mth in list(set([x[0] for x in simu_runs[1]])):
        cis_final = []
        rrr = [x for x in simu_runs[1] if x[0]==mth]
        final_comp=[]
        try:
            for x in rrr:
                if x[1][:2]=="CV":
                    a1,a2=x[2][0]
                    final_comp.append([x[0]+x[1],nse(a1, a2),rmse(a1,a2),1/(2-nse(a1, a2)),kge(a1, a2)])
                else:
                    pass
            if len(final_comp)==ncvs:
                cis_final.append(["CV-Best-"+mth,final_comp,CI_Scipy(np.array(final_comp)[:,1].astype(float)),
                                  CI_Scipy(np.array(final_comp)[:,2].astype(float)),
                                  CI_Scipy(np.array(final_comp)[:,3].astype(float)),
                                  CI_Scipy(np.array(final_comp)[:,4].astype(float)),
                        min_max_med_var(np.array(final_comp)[:,1].astype(float)),
                                  min_max_med_var(np.array(final_comp)[:,2].astype(float)),
                                  min_max_med_var(np.array(final_comp)[:,3].astype(float)),
                                 min_max_med_var(np.array(final_comp)[:,4].astype(float))])
            else:
                pass
        except:
            pass
        if len(cis_final)==0:
            pass
        else:
            outpvf = cis_final[0]

            line_ind_cv = outpvf[0].split("_")[-1]
            cv_evals = np.array(outpvf[1])
            ci_nse = outpvf[2]
            ci_rmse = outpvf[3]
            ci_nse_norm = outpvf[4]
            ci_kge = outpvf[5]
            mmm_nse = outpvf[6]
            mmm_rmse = outpvf[7]
            mmm_nse_norm = outpvf[8]
            mmm_kge = outpvf[9]
            df = pd.DataFrame(cv_evals)
            df.columns = ["Partition","nse","rmse","nse_norm","kge"]
            dfs.append([line_ind_cv,df])
            cis_final = []
            try:
                for x in rrr:
                    if x[1][:2]!="CV":
                        a1,a2=x[2][0]
                        cis_final.append(["Holdout-Best-"+mth,a1,a2,nse(a1, a2),rmse(a1,a2),1/(2-nse(a1, a2)),kge(a1, a2)])
                    else:
                        pass
            except:
                pass
            if len(cis_final)==0:
                pass
            else:
                outpvf = cis_final[0]
                line_ind = outpvf[0].split("_")[-1]
                pred_v = np.array(outpvf[1])
                ori_v = np.array(outpvf[2])
                nse_h = outpvf[3]
                rmse_h = outpvf[4]
                nse_norm_h = outpvf[5]
                kge_h = outpvf[6]
                df2 = pd.DataFrame([pred_v,ori_v])
                df2.columns = cols
                df2.index = [line_ind,"Holdout"]
                dfs2.append(df2)
                df3 = pd.DataFrame([[nse_h,rmse_h,nse_norm_h,kge_h]+mmm_nse+ci_nse+mmm_rmse+ci_rmse+mmm_nse_norm+ci_nse_norm+mmm_kge+ci_kge])
                df3.columns = ["NSE_Holdout","RMSE_Holdout","NSE_Norm_Holdout","KGE_Holdout"]+[x+"NSE" for x in ["Min","Max","Median","Var"]]+["CI-NSE"+x for x in ["Lower","Mean",
                              "Upper"]]+[x+"RMSE" for x in ["Min","Max","Median","Var"]]+["CI-RMSE"+x for x in ["Lower","Mean",
                              "Upper"]]+[x+"NSE_Norm" for x in ["Min","Max","Median","Var"]]+["CI-NSE_norm"+x for x in ["Lower",
                            "Mean","Upper"]]+[x+"KGE" for x in ["Min","Max","Median","Var"]]+["CI-KGE"+x for x in ["Lower","Mean","Upper"]]
                df3.index = [line_ind_cv]
                dfs3.append(df3)
        aaa,bbb,ccc = [dfs,dfs2,dfs3]
    with pd.ExcelWriter(root+'\\Summary.xlsx') as writer:
        final_eval = pd.concat(bbb)
        final_eval = final_eval[~final_eval.index.duplicated(keep='last')]
        final_eval.to_excel(writer, sheet_name='Predictions')
        final_stats = pd.concat(ccc).loc[final_eval.index[:-1]]
        final_stats.to_excel(writer, sheet_name='Statistics')
        for dfi in aaa:
            if dfi[0] in list(final_eval.index):
                dfi[1].to_excel(writer, sheet_name=dfi[0])
            else:
                pass
            
def prepare_dl_algorithms(algorithms, input_size):

    # In the ML4FF framework, besides estimators from hpsklearn we also considered DL estimators of the LSTM family. These are incorporated into the framework by taking adavantaged of the skorch package, which relies on PyTorch.
    # In order to use the skorch package, we need to define the Neural Network Regressors to be used, which can be created based on classes of models.
    # For reproducibility, we set the random seed of PyTorch. It is worth highlighting that completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms. Furthermore, results may not be reproducible between CPU and GPU executions, even when using identical seeds.

    net_LSTM = NeuralNetRegressor(
        LSTMModel,
        module__input_size=input_size,
        max_epochs=200,
        verbose=0,
        criterion=torch.nn.MSELoss,
        batch_size=128,
        train_split=InOrderSplit(0.1),
        optimizer = torch.optim.Adam,
        warm_start=False,
        callbacks=[EarlyStopping(monitor='valid_loss',patience=15,threshold=0.00001)]
    )

    net_LSTMPreModel = NeuralNetRegressor(
        LSTMPreModel,
        max_epochs=200,
        verbose=0,
        criterion=torch.nn.MSELoss,
        batch_size=64,
        train_split=InOrderSplit(0.1),
        optimizer = torch.optim.Adam,
        warm_start=False,
        callbacks=[EarlyStopping(monitor='valid_loss',patience=15,threshold=0.00001)]
    )
        
    # After defining the regressors whose hyperparameters will be tuned, it is important to define the hyperopt search space. These are presented below for both the LSTM models considered.

    ########################################################################################
    ################## Insert your customizations in the DL methods here! ##################
    ########################################################################################
        
    paramsLSTM = {
        'module__dropout': hp.uniform('module__dropout', 0.2, 0.5),
        'module__hidden_size': hp.choice('module__hidden_size',[1,5,10,20,25]),
        'module__num_layers': hp.choice('module__num_layers',[2,3,4,5])
    }

    paramsLSTMPreModel = {
        'batch_size': hp.choice('batch_size',[32,64,128,256]),
        'module__dropout': hp.uniform('module__dropout', 0.2, 0.5),
        'module__hidden_size': hp.choice('module__hidden_size',[1,5,10,20,25]),
        'module__num_layers': hp.choice('module__num_layers',[2,3,4,5]),
        'module__num_units': hp.choice('module__num_units',[10,20,30,40]),
        'module__nonlin': hp.choice('module__nonlin',[F.leaky_relu,F.celu]),
    }
    
    all_algorithms_DL =[
    ["DL_LSTM",net_LSTM,paramsLSTM],
    ["DL_LSTMPre",net_LSTMPreModel,paramsLSTMPreModel],
    ]

    algorithms = [item.lower() for item in algorithms]

    if not set(algorithms).issubset([item[0].lower() for item in all_algorithms_DL]):
        raise ValueError(f"Some specified DL algorithms are either unavailable in the ML4FF Framework or incorrectly spelled. Please revise the following algorithms: {set(algorithms) - set([item[0].lower() for item in all_algorithms_DL])}")

    # Similarly for models from hpsklearn, we build the list of candidate models for DL:

    all_candidates_DL = [item for item in all_algorithms_DL if item[0].lower() in algorithms]

    return all_candidates_DL

def prepare_ml_algorithms(algorithms):

    all_algs = [[x.name,x,buil_dict(x)] for x in all_regressors("reg").inputs()]

    #From the list "all_algs", build the reduced list containing only the ML methods of interest to ML4FF:

    all_algorithms_ML = ["sklearn_RandomForestRegressor","sklearn_BaggingRegressor","sklearn_GradientBoostingRegressor",
               "sklearn_LinearRegression","sklearn_BayesianRidge","sklearn_ARDRegression","sklearn_LassoLars",
               "sklearn_LassoLarsIC","sklearn_Lasso","sklearn_ElasticNet","sklearn_LassoCV",
               "sklearn_TransformedTargetRegressor","sklearn_ExtraTreeRegressor","sklearn_DecisionTreeRegressor",
               "sklearn_LinearSVR","sklearn_PLSRegression","sklearn_MLPRegressor","sklearn_DummyRegressor",
               "sklearn_TheilSenRegressor","sklearn_OrthogonalMatchingPursuitCV","sklearn_OrthogonalMatchingPursuit",
               "sklearn_RidgeCV","sklearn_Ridge","sklearn_SGDRegressor","sklearn_PoissonRegressor",
               "sklearn_ElasticNetCV",
               "sklearn_KNeighborsRegressor","sklearn_RadiusNeighborsRegressor",
               "sklearn_XGBRegressor","sklearn_GaussianProcessRegressor","sklearn_NuSVR","sklearn_LGBMRegressor"
               ]
    
    algorithms = [item.lower() for item in algorithms]
    all_algorithms_ML = [item.lower() for item in all_algorithms_ML]
    
    if not set(algorithms).issubset(all_algorithms_ML):
        raise ValueError(f"Some specified ML algorithms are either unavailable in the ML4FF Framework or incorrectly spelled. Please revise the following algorithms: {set(algorithms) - set(all_algorithms_ML)}")

    all_candidates_ML = [item for item in all_algs if item[0].lower() in algorithms]

    return all_candidates_ML

def reset_seed(seed_value):
    """
    Resets the random seed for various libraries to ensure reproducibility across ML and DL experiments.
    """
    # 1. Set PYTHONHASHSEED for reproducible hashing (must be set at program start)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    
    # 2. Seed Python's built-in random module
    random.seed(seed_value)
    
    # 3. Seed NumPy's random generator
    np.random.seed(seed_value)
    
    # 4. Seed PyTorch for CPU and GPU computations
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
        
        # Ensure deterministic behavior for CUDA (if using cuDNN)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 5. Enforce deterministic algorithms in PyTorch (for operations that support it)
    torch.use_deterministic_algorithms(True, warn_only=True)

def execute_ml4ff(config: Config):  

    datasetHandler = DatasetHandler(config)

    all_candidates_ML = prepare_ml_algorithms(config.ml_algorithms)
    all_candidates_DL = prepare_dl_algorithms(algorithms=config.dl_algorithms, input_size=datasetHandler.get_input_size())
    # It is important to notice that for DL models we added the prefix "DL_" to their names, which is important for the subsequent calculations of ML4FF.
    # Finally, the list of all methods of interest is generated by concatenating all_candidates_ML and all_candidates_DL.
    all_candidates = all_candidates_ML + all_candidates_DL

    # Run the Nested-CV.
    run_ML4FF = RefineNCV_General(datasetHandler,config.seed,config.inner_cv,config.outer_cv,all_candidates, config.result_path, config.save_models)

    #ML4FF_pickled = buil_results_pkl(all_candidates, root+"\\Models")

    perf_excel(run_ML4FF, config.result_path)

    build_excel(datasetHandler.get_dataset(), config.result_path, config.outer_cv, config.holdout_slice, run_ML4FF)

# Main execution
if __name__ == "__main__":
    # Configuring the argparse
    parser = argparse.ArgumentParser(description="ML4FF Framework")
    
    # Adding arguments
    parser.add_argument("-c", "--config",
                        required=True,
                        help="Configuration file. Please see an example at https://github.com/jaqueline-soares/ML4FF-framework.")
        
    # Parsing the arguments
    args = parser.parse_args()

    config = Config.from_json(args.config)    
        
    execute_ml4ff(config)