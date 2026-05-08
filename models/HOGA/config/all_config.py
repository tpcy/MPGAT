from ml_collections import ConfigDict
import torch

from config.ds_config import * 
from config.model_config import * 


def all_config():
    config = ConfigDict()
    config.device = torch.device('cuda:0')

    config.cora = cora_config()
    config.pubmed = pubmed_config()
    config.citeseer = citeseer_config()
    config.computers = computers_config()
    config.photo = photo_config()
    config.wisconsin = wisconsin_config()
    config.texas = texas_config()

    config.experiments = ConfigDict()
    config.experiments.num_repeats = 20 

    config.baselines = ConfigDict()
    config.baselines.names = ['HoGA_GAT']
    config.baselines.HoGA_GAT = hoga_gat_config()


    return config 
