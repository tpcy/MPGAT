# third party 
from itertools import product
import matplotlib.pyplot as plt
import yaml
from torch.optim.lr_scheduler import ExponentialLR

# first party 
#from grand_src.run_GNN import train 
from utils import *


def get_val_loss(model, data, loss, device) -> float:  
    """
        Gets the validation loss for the model
    """
                                      
    model.eval()                                                                    
    args, labels, masks = get_args(data, device)
    pred = model(**args)

    val_loss = loss(pred[masks['val']], labels[masks['val']]).item()

    return val_loss


def train_step_normal(model, optimiser, data, device=torch.device('cpu'), lf=torch.nn.CrossEntropyLoss()):
    model.train()
    optimiser.zero_grad()                  

    args, labels, masks = get_args(data, device)
    pred = model(**args)
    
    tl = lf(pred[masks['train']], labels[masks['train']])

    tl.backward()
    optimiser.step()

    return tl.item()


def test(model, data, device):
    model.eval()
    args, labels, masks = get_args(data, device)
    pred = model(**args).argmax(dim=-1)

    accs = []
    for mask in masks.values():
        accs.append(int((pred[mask] == labels[mask]).sum()) / int(mask.sum()))

    return accs


def train_model(
                    model_config, 
                    model, 
                    data, 
                    loss, 
                    save_dir, 
                    device, 
                    step_method='normal', 
                    do_save=True
                ):
    
    """
        Trains the model, saves state to checkpoint & returns train and validation loss
    """

    if step_method == 'normal':
        train_step = train_step_normal 
    elif step_method == 'grand':
        train_step = train 
    else:
        raise ValueError(f"Invalid step method: {step_method}")
 
    num_epochs = model_config.training.num_epochs
    save_freq = model_config.training.save_freq
    weight_decay = model_config.training.weight_decay           
    lr = model_config.training.lr 
    gamma = model_config.training.decay 

    if model_config.training.optimizer == 'adam':
        optimiser = torch.optim.Adam(model.parameters(), weight_decay=weight_decay, lr=lr)
    else:
        raise ValueError(f"Invalid optimiser name {model_config.training.optimiser} in configuration file")

    scheduler = ExponentialLR(optimiser, gamma=gamma)

    best_val_acc = best_tea = 0
    val_loss, train_loss = [], []

    model = model.to(device)
    for epoch in range(num_epochs):        
        tl = train_step(model, optimiser, data, device=device, lf=loss)
        vl = get_val_loss(model, data, loss, device) 

        train_loss.append(tl)
        val_loss.append(vl)

        scheduler.step()

        train_acc, val_acc, test_acc = test(model, data, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_tea = test_acc
            save_to_checkpoint(model, save_dir)

        if epoch % save_freq == 0 or epoch == 0: 
            print('Epoch:', epoch, 'Train Loss', round(tl, 4), 'Val Loss', round(vl, 4), 'Train Acc:', round(train_acc, 4), 
            'Val Acc:', round(val_acc, 4), 'Test Acc:', round(test_acc, 4), 'Best Val Acc:', round(best_val_acc, 4), 
            'Best Test Acc:', round(best_tea, 4))

    return train_loss, val_loss

def meta_train(
                    config: ConfigDict,
                    ds_config: ConfigDict,
                    models: Dict[AnyStr, List[nn.Module]],
                    dataset: Dict[AnyStr, Any], 
                    loss
                ) -> Dict:

    """
        Handles the repetition of experiment runs
    """

    num_repeats = config.experiments.num_repeats
    device = config.device

    metric_callables = get_metric_functions(ds_config, device)
    meta_metrics = {mn: {} for mn in models}

    for n, (mn, model) in product(range(num_repeats), models.items()):  
        if mn in ['GRAND', 'MultiHop_GRAND', 'HiGCN', 'GraphTransformer', 'SPAGAN', 'MixHop']:               # reset parameters not well supported in GRAND source 
            model = model() 
        else:
            model.reset_parameters() 
        model = model.to(device)
        
        full_path = os.path.join('checkpoints', ds_config.name, mn+str(n))
        if not os.path.exists(full_path):
            os.makedirs(full_path)

        if mn in ['MultiHop_GAT', 'MultiHop_GRAND']:
            data = dataset['multihop_dataset']
        else:
            data = dataset['dataset']

        if mn in ['GRAND', 'MultiHop_GRAND']:
            step_method = 'grand'
        else:
            step_method = 'normal'

        model_config = config.baselines[mn] 
        train_loss, val_loss = train_model(model_config, model, data, loss, full_path, device, step_method)

        model = torch.load(os.path.join(full_path, 'model.pt'), weights_only=False)
        metrics = collect_metrics(model, data, metric_callables, device)
        
        # save losses to figure
        plt.figure()
        plt.plot(train_loss)
        plt.plot(val_loss)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.xticks([n+1 for n in range(len(train_loss)) if (n+1) % 10 == 0 or n == 0])
        plt.xlim([1, len(train_loss)])
        plt.title('Training Curves')
        plt.legend(['train', 'val'])
        plt.savefig(os.path.join(full_path, 'loss_curves.pdf'))
        plt.close()

        # save metrics
        np.savetxt(os.path.join(full_path, 'metrics.txt'), [[str(mem), str(val)] for mem, val in metrics.items()], fmt='%s')
        meta_metrics = join_metrics(meta_metrics, metrics, mn)

        # write configuration file
        config_dict = config.to_dict()
        del config_dict['device']   # not serializable
        with open(os.path.join(full_path, 'config.yml'), 'w') as writer:
            yaml.dump(config_dict, writer)

    return meta_metrics
