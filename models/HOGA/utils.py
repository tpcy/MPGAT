# get_metric_functions, fetch_dataset, create_models, collect_metrics, assign_to_config, best_params_dict
# best_params_dict could be here 

# third party 
import numpy as np 
import networkx as nx
import os 
 
import torch_geometric.datasets as datasets
import torch
import torch.nn as nn
from ml_collections import ConfigDict
from torch_geometric.typing import Tensor, Adj
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.transforms import Compose, NormalizeFeatures, RandomNodeSplit
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.nn.models import GCN, GAT
from torchmetrics import Accuracy, AUROC

from copy import copy
from functools import partial 
from typing import *

# first party 
from multi_hop import GenericGAT
#from grand_src import merge_cmd_args
from hop_utils import get_K_adjs


class CombinedLoader:
    def __init__(self, *loaders):
        self.length = np.Inf
        self.iters = []
        self.loaders = loaders[0]
        self.dataset = self.loaders[0].dataset 

        for loader in self.loaders:
            self.length = min(self.length, len(loader))

    def gen(self, loader, edge_set):
        for item in loader:
            item.edge_index = edge_set     
            yield item

    def __iter__(self):
        self.iters = [] 
        for loader in self.loaders:
            gen = self.gen(loader, loader.dataset.edge_index)
            self.iters.append(gen)

        return self

    def __next__(self):
        items = [] 
        for n, it in enumerate(self.iters):
            next_item = next(it)
            items.append(next_item)

        return self.length
        

def to_numpy(x: Tensor):
    return x.cpu().detach().numpy()


def get_ds_config(config: ConfigDict, ds_name: str) -> ConfigDict:
    if ds_name == 'Cora':
        ds_config = config.cora 
    elif ds_name == 'Pubmed':
        ds_config = config.pubmed
    elif ds_name == 'Citeseer':
        ds_config = config.citeseer
    elif ds_name == 'Computers':
        ds_config = config.computers
    elif ds_name == 'Photo':
        ds_config = config.photo 
    elif ds_name == 'Actor':
        ds_config = config.actor 
    elif ds_name == 'Wisconsin':
        ds_config = config.wisconsin 
    elif ds_name == 'Texas':
        ds_config = config.texas
    else:
        raise ValueError('Invalid dataset name')

    return ds_config


def get_metric_functions(ds_config, device):
    metric_names = ds_config.metrics
    callables = {}
    for mn in metric_names:
        if mn == 'Accuracy':
            callables[mn] = Accuracy(task="multiclass", num_classes=ds_config.num_classes).to(device)
        elif mn == 'AUC':
            callables[mn] = AUROC(task="multiclass", num_classes=ds_config.num_classes).to(device)
        else:
            raise ValueError()

    return callables


def load_from_checkpoint(config: ConfigDict, ds_name: str, nrepeats: int) -> Dict[AnyStr, Any]:   
    """
        Loads in the metrics of models from checkpoint 
    """

    model_names = config.baselines.names
    metric_names = config[ds_name.lower()].experiments.metrics
    metric_dict = {mn: {key: [] for key in metric_names} for mn in model_names}

    for mn, mi in product(model_names, range(nrepeats)):                                
        full_path = os.path.join('checkpoints', ds_name, mn+str(mi), 'metrics.txt')
        if not os.path.exists(full_path):
            print(f'Metric file path {full_path} not found')
            continue
        # read into metric dict 
        with open(full_path, 'r') as reader:
            for mline in reader.readlines():
                mline = mline.split(' ')
                key = str(mline[0])
                val = float(mline[1])
                metric_dict[mn][key].append(val) 

    return metric_dict 


def save_to_checkpoint(model: nn.Module, save_dir: str):
    """
        Saves the model
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(model, os.path.join(save_dir, 'model.pt'))


def get_args(batch, device):
    features, labels = batch.x.to(device), batch.y.to(device)
    masks = {'train': batch.train_mask, 'val': batch.val_mask, 'test': batch.test_mask}
    call_args = {'x': features, 'edge_index': batch.edge_index[0]}
    
    return call_args, labels, masks 


def collect_metrics(model, data, callables: Dict[AnyStr, Callable], device) -> Dict[AnyStr, float]:
    """
        Collects metrics from a series of callable metric functions
    """
    
    metrics = {kw: 0 for kw in callables}
    model.eval()

    call_args, labels, masks = get_args(data, device)
    preds = model(**call_args)

    for it_nm, item_func in callables.items():
        stat = to_numpy(item_func(preds[masks['test']], labels[masks['test']]))
        metrics[it_nm] = stat 

    return metrics
    

def join_metrics(existing_dict: Dict[AnyStr, List], new_dict: Dict[AnyStr, float], model_name: str):
    """
        Adds the new metric dictionary to a collection of metrics of all previous runs
    """

    for sn in new_dict:
        if sn not in existing_dict[model_name]:
            existing_dict[model_name][sn] = []
        existing_dict[model_name][sn].append(new_dict[sn])

    return existing_dict


def assign_to_config(config, opt, training=True):
    for key in opt:
        best_param = opt[key]
        if isinstance(best_param, ConfigDict):
            print(key)
            continue 
        if key in config and config[key] is not None:
            del config[key]

        config[key] = best_param 

    if hasattr(opt, 'decay') and training:
        config.training.weight_decay = opt['decay'] 
        config.training.lr = opt['lr']
        config.dropout = opt['dropout']
    
    
def create_models(config: ConfigDict, ds_config: ConfigDict, data_dict) -> Dict[AnyStr, List[nn.Module]]:
    """
        Given some dataset, will create number and type of models specified in the config file, packed into a dictionary
    """

    in_chnls = ds_config.in_channels        # data
    hdn_chnls = ds_config.hidden_channels
    num_classes = ds_config.num_classes
    ds_name = ds_config.name
    dataset = data_dict['dataset']
    num_nodes = data_dict['num_nodes']
    device = config.device

    # create model
    baselines = {}
    for mn in config.baselines.names:
        if mn == 'GAT':
            model_config = config.baselines.GAT
            dropout = model_config.drop_out
            num_layers = model_config.num_layers
            num_heads = model_config.num_heads 

            #model =  GenericGAT(config.baselines.GAT, ds_config, device=device, layer_type='normal')
            model = GAT(in_chnls, hdn_chnls, out_channels=num_classes, heads=8)

        elif mn == 'GCN':
            model_config = config.baselines.GCN
            dropout = model_config.drop_out 
            
            model = GCN(in_chnls, hdn_chnls, num_layers=model_config.num_layers, out_channels=num_classes, dropout=dropout)

        elif mn == 'GRAND':   
            best_opt = best_params_dict[ds_config.name]
            model_config = config.baselines.GRAND
            assign_to_config(model_config, best_opt)

            model = lambda: GNN(model_config, dataset, device)
        
        elif mn == 'HoGA_GRAND':
            best_opt = best_params_dict[ds_config.name]
            model_config = config.baselines.HoGA_GRAND

            assign_to_config(model_config, config.baselines.GRAND, training=False)
            assign_to_config(model_config, best_opt)

            model_config['block'] = 'attention'
            dataset = data_dict['multihop_dataset']

            dataset.edge_index = dataset.data.edge_index = dataset.edge_index[0][0]

            model = lambda: GNN(model_config, dataset, device)

        elif mn == 'HoGA_GAT':
            num_nodes = data_dict['num_nodes']
            model = GenericGAT(config.baselines.HoGA_GAT, ds_config, device=device, num_nodes=num_nodes, layer_type='multi_hop')
        
        else:
            raise ValueError('Invalid/unsupported model name')

        baselines[mn] = model

    return baselines
    

def gen(num_layers, num_heads):
    for layer_idx in range(num_layers):
        heads = num_heads[layer_idx]
        for head_idx in range(heads):
            yield layer_idx, head_idx 


def mask_transform(data, **masks):
    data.train_mask = masks['train']
    data.val_mask = masks['val']
    data.test_mask = masks['test']

    return data


def edge_index_to_nx(edge_index):
    graph = nx.Graph()
    edges = edge_index.t().tolist()
    graph.add_edges_from(edges)

    return graph

def index_mask(ds, i=0):
    ds.train_mask = ds[0].train_mask[:,i]
    ds[0].train_mask = ds[0].train_mask[:,i]
    ds.val_mask = ds[0].val_mask[:,i]
    ds[0].val_mask = ds[0].val_mask[:,i]
    ds.test_mask = ds[0].test_mask[:,i]
    ds[0].test_mask = ds[0].test_mask[:,i]


def fetch_dataset(config: ConfigDict, ds_name: str, unpack: bool = False, add_args=dict({})) -> Union[Dict[AnyStr, Any], List]:
    """
        Downloads (if not present at path) a particular dataset, and returns unpacked dictionary
    """

    device = config.device
    ds_path = os.path.join('Datasets', ds_name) 

    train_path, val_path, test_path = os.path.join(ds_path, 'train_mask'), os.path.join(ds_path, 'val_mask'), os.path.join(ds_path, 'test_mask')
    create_masks = (np.sum([os.path.exists(train_path), os.path.exists(val_path), os.path.exists(test_path)]) != 3)
    
    transform = [NormalizeFeatures()]
    if not create_masks:        
        train_mask, val_mask, test_mask = torch.load(train_path), torch.load(val_path), torch.load(test_path)
        transform.append(partial(mask_transform, train=train_mask, val=val_mask, test=test_mask))
    transform = Compose(transform)                  

    if ds_name in ['Pubmed', 'Cora', 'Citeseer']:   # fixed split datasets 
        ds = datasets.Planetoid(root=ds_path, name=ds_name, transform=NormalizeFeatures()).to(device)
    elif ds_name in ['Photo', 'Computers']:
        ds = datasets.Amazon(root=ds_path, name=ds_name, transform=transform).to(device)
    elif ds_name in ['Actor']:
        ds = datasets.Actor(root=ds_path, transform=transform).to(device)
    elif ds_name in ['Texas', 'Wisconsin']:
        ds = datasets.WebKB(root=ds_path, name=ds_name).to(device)
        index_mask(ds)
    else:   
        raise ValueError('Invalid dataset name')  

    if ds_name not in ['Texas', 'Wisconsin']:
        ds.train_mask = ds[0].train_mask
        ds.val_mask = ds[0].val_mask
        ds.test_mask = ds[0].test_mask

    torch.save(ds[0].train_mask, train_path)
    torch.save(ds[0].val_mask, val_path)
    torch.save(ds[0].test_mask, test_path)

    ds_config = get_ds_config(config, ds_name)
    num_classes = ds_config.num_classes
    model_names = config.baselines.names

    data = {
                'dataset': ds,
                'graph': ds.edge_index,
                'train': ds[0].train_mask,
                'test': ds[0].test_mask,
                'val': ds[0].val_mask,
                'num_classes': num_classes,  
                'num_features': ds[0].num_features, 
                'num_nodes': ds_config.num_nodes,
                'name': ds_config.name 
            }

    if 'HoGA_GAT' in model_names or 'HoGA_GRAND' in model_names:       
        model_config = config.baselines.HoGA_GAT
        K_hop = model_config.K_hops
        load_samples = model_config.load_samples

        num_heads = model_config.num_heads 
        num_layers = model_config.num_layers 

        data['multihop_dataset'] = [[] for _ in range(num_layers)] 
        for layer_idx, head_idx in gen(num_layers, num_heads):
            if layer_idx >= 1:                                                 # hard code since I know only ony adj list is needed 
                continue 
            nlayer_heads = num_heads[layer_idx]
            save_path = os.path.join(ds_config.save_path, model_config.select_method, str(layer_idx), str(head_idx))
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            if load_samples:
                edges = [] 
            else:
                edges = get_K_adjs(
                                    ds.edge_index,
                                    model_config, 
                                    ds_config, 
                                    feature_set=ds.x.to(device), 
                                    device=device
                ) 

            # populate for embedding component of model
            for k in range(K_hop):
                if k == 0:
                    if load_samples:
                        edges.append(ds.edge_index.to(torch.int64))
                    continue 
                path = os.path.join(os.getcwd(), save_path, str(k))
                #print('save path:', path)
                if load_samples:
                    edges.append(torch.load(path).to(device).to(torch.int64))
                else:
                    torch.save(edges[k].to(torch.int64), os.path.abspath(path))

            multihop_data, num_nodes = [], []
            for k, edge_set in enumerate(edges):    
                multihop_data.append(edge_set[:,0:edges[0].shape[-1]])       

            data['multihop_dataset'][layer_idx].append(multihop_data)

        for l in range(len(data['multihop_dataset'])):
            if l == 0:
                continue 
            headl = min(num_heads[l], num_heads[0])
            data['multihop_dataset'][l] = data['multihop_dataset'][0][0:headl]
        
        cp_dataset = copy(data['dataset'])
        cp_dataset.edge_index = cp_dataset.data.edge_index = data['multihop_dataset']
        data['multihop_dataset'] = cp_dataset

    data = ConfigDict(data)

    if unpack:
        return [data[key] for key in data]
    else:
        return data

