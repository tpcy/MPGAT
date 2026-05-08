# third party 
import os 
import sys 

directory_path = os.path.abspath(os.getcwd())

if directory_path not in sys.path:
    sys.path.append(directory_path)

from itertools import product 
from ml_collections import ConfigDict
import copy 
import argparse
import numpy as np
import json 
import torch
import torch.nn as nn 



# first party 
from config.all_config import graph_config 
from utils import *
from train import train_model


def minor_parameters(method: str):
    parameters_dict = { 
                            'lr': [0.01],                   
                            'drop_out': [0.60],                 
                            'gamma': [1.0],                   
                            'weight_decay': [0.005],
                            'agg_func': ['mean'],                     
                            'num_epochs': [300]
    }

    return list(parameters_dict.keys()), list(parameters_dict.values())


def small_metric_join(metric_dict, outcome):
    for stat_nm, stat_val in outcome:
        if stat_nm in metric_dict:
            metric_dict[stat_nm].append(stat_val)
        else:
            metric_dict[stat_nm] = [stat_val]


def alter_config(model_settings, run_config):
    for key, val in run_config.items():
        if key in ['lr', 'weight_decay', 'gamma', 'num_epochs']:  
            model_settings.training[key] = run_config[key]
        else:
            model_settings[key] = run_config[key]

            
def tune_hp(config, model_config, ds_config, method, project, num_repeats, model_nm, expr_nm):
    """
        Tune hyperparameters of individual model 
    """

    def train(_model, run_hp):              # maybe get multiple repeats going in here and move with statement outside? 
        accuracies = [] 

        alter_config(model_config, run_hp)

        device = config.device 
        metric_callables = get_metric_functions(ds_config, device)

        if 'HoGA' in model_nm:
            dataset = data_dict['multihop_dataset']  
        else:
            dataset = data_dict['dataset']

        if 'GRAND' in model_nm:
            step_method = 'grand'
        else:
            step_method = 'normal'

        loss = nn.CrossEntropyLoss()                                            # assuming a classification task        
        
        # get averaged accuracy 
        for n in range(num_repeats):  # here is changed 
            save_dir = os.path.join(f'experiments/abalation_test/checkpoints', f'{ds_config.name}_{expr_nm}', model_nm)  
            if hasattr(_model, 'reset_parameters'):
                _model.reset_parameters()
                model = _model 
            else:
                model = _model()                                                        # changed for GRAND
            train_model(model_config, model, dataset, loss, save_dir, device, step_method=step_method)
            model = torch.load(os.path.join(save_dir, 'model.pt'))
            model_run = collect_metrics(model, dataset, metric_callables, device)
            accuracies.append(model_run['Accuracy'])

        return accuracies

    hp_names, hp_vals = minor_parameters(method)
    config.baselines[model_nm] = model_config 

    # hard code 
    if model_nm == 'HoGA_GRAND':
        config.baselines.HoGA_GAT.K_hops = model_config.K_hops 

    data_dict = fetch_dataset(config, ds_config.name)

    model = create_models(config, ds_config, data_dict)[model_nm]

    global_mean, global_std, best_config = 0, -1, None 
    for run_hp in product(*hp_vals):    
        run_hp = {key: val for key, val in zip(hp_names, run_hp)}
        
        accs = train(model, run_hp)
        acc_mean = np.mean(accs)
        acc_std = np.std(accs)
        
        print(run_hp)
        print(ds_config.name)
        print(accs)
        print(np.mean(accs), np.std(accs))
        print(model_config.num_heads)
        print()

        if acc_mean > global_mean:
            global_mean = acc_mean
            global_std = acc_std  
            best_config = run_hp 
    
    return best_config, global_mean, global_std


def run(pargs): 
    # unpack 
    dataset = pargs.dataset 
    project = pargs.project 
    p_nms, p_vls = pargs.parameter, pargs.values
    nheads = pargs.nheads
    nlayers = pargs.nlayers 
    khops = pargs.khops
    method = pargs.method 
    num_repeats = pargs.nrepeats
    gpu = pargs.gpu 

    config = graph_config()
    ds_config = config[dataset.lower()]
    if gpu is not None and gpu in [0, 1]: 
        config.device = torch.device(f'cuda:{gpu}')
    device = config.device 
    metric_callables = get_metric_functions(ds_config, device)

    # choose which model to use 
    if np.sum([pargs.ho_GAT, pargs.norm_GRAND, pargs.norm_GAT, pargs.ho_GRAND]) > 1:
        raise ValueError('Only one model may be specfied at a time')
    elif pargs.norm_GAT:
        model_nm = 'GAT'
        config.baselines.names = ['GAT']
    elif pargs.ho_GAT:
        model_nm = 'HOGA_GAT'
        config.baselines.names = ['HOGA_GAT']
    elif pargs.ho_GRAND:
        model_nm = 'HOGA_GRAND'
        config.baselines.names = ['HOGA_GRAND']
    elif pargs.norm_GRAND:
        model_nm = 'GRAND'
        config.baselines.names = ['GRAND']


    model_config = config.baselines[model_nm]
    if model_nm in ['GRAND', 'HOGA_GRAND']:
        best_opt = best_params_dict[ds_config.name]
        assign_to_config(model_config, config.baselines.GRAND, training=False)
        assign_to_config(model_config, best_opt)
    #model_config.select_method = ''

    # only supports one 
    abs_results = []
    if nheads > 0:
        p_nms = ['num_heads']
        p_vls = [
                    [[i+1, j+1] for (i, j) in product(range(nheads), range(nheads))]
                ]
        nm = 'heads'
    elif nlayers > 0:
        p_nms = ['num_layers', 'num_heads']
        p_vls = [
                    [l+1 for l in range(nlayers)],
                    [[8 if hi == 0 else 1 for hi in range(l+1)] for l in range(nlayers)]
                ]
        gen = zip(*p_vls)
        nm = 'layers'
    elif khops > 0:
        p_nms = ['K_hops']
        gen = p_vls = [
                    [k+1] for k in range(2, khops)
                ]
        nm = 'khop'
    else:
        p_nms = ['num_layers']
        p_vls = [2]
        nm = 'hp'

    # set up save dir 
    ab_parent_dir = os.path.join('experiments/abalation_test/checkpoints', f'{dataset}_{nm}') 
    if not os.path.exists(ab_parent_dir):
        os.makedirs(ab_parent_dir)
    
    for val_set in gen:
        val_set = {key: val for key, val in zip(p_nms, val_set)}

        cpy_config = copy.deepcopy(config)
        cpy_config.baselines.names = [model_nm]
        model_config = cpy_config.baselines[model_nm]
        alter_config(model_config, val_set)
 
        output_str = model_nm+'_'                           
        for nm, val in val_set.items():          
            output_str += f'{nm}_{val}_'            
            model_config[nm] = val                  
        output_str = output_str[:-1]                 

        setting, mean, std = tune_hp(config, model_config, ds_config, method, project+output_str, num_repeats, model_nm, nm)

        for key, val in val_set.items():
            setting[key] = val 

        file_abs = os.path.join(ab_parent_dir, output_str)          
        with open(file_abs+'.txt', 'w') as writer:
            writer.write(f'mean: {mean}, std: {std}')
        with open(file_abs+'.json', 'w') as json_file:
            json.dump(setting, json_file, indent=4)

        abs_results.append(mean)
    
    return abs_results 

args = argparse.ArgumentParser()
args.add_argument('--dataset', type=str, help='The name of the dataset you wish to run')
args.add_argument('--nrepeats', type=int, help='Number of repetitions')
args.add_argument('--gpu', type=int, help='Number of repetitions', default=None)
args.add_argument('--nheads', type=int, help='Number of heads', default=0)
args.add_argument('--nlayers', type=int, help='Number of heads', default=0)
args.add_argument('--khops', type=int, help='Number of heads', default=0)
args.add_argument('--parameter', type=str, help='The name of the parameter(s) you wish to alter', default=None)
args.add_argument('--project', help='Name of project', default='test')
args.add_argument('--method', help='Method from which to conduct the HP search', default='grid')
args.add_argument('--values', type=list, help='The value of the parameter(s) to range over', default=None)
args.add_argument('--ho_GAT', help='Wether to use GAT as a backbone', action='store_true', default=False)
args.add_argument('--norm_GAT', help='Wether to use the normal GAT as a backbone', action='store_true', default=False)
args.add_argument('--norm_GRAND', help='Wether to use GRAND as a backbone', action='store_true', default=False)
args.add_argument('--ho_GRAND', help='Wether to use GRAND as a backbone', action='store_true', default=False)

pargs = args.parse_args()

run(pargs) 