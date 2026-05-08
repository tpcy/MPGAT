# third party 
import torch.nn as nn
import torch
try:
    import torch_sparse
except Exception:
    torch_sparse = None

import torch_geometric.nn as tgnn 
from torch_geometric.utils import dropout_adj, softmax 

# first party 
from hop_utils import *
#from grand_src.function_transformer_attention import SpGraphTransAttentionLayer


class GenericGAT(nn.Module):
    def __init__(self, model_config, ds_config, device, layer_type='normal', **kwargs):
        super(GenericGAT, self).__init__()

        num_layers = model_config.num_layers
        input_dim = ds_config.in_channels
        output_dim = ds_config.num_classes
        hidden_dim = ds_config.hidden_channels
        dropout = model_config.drop_out
        num_heads = model_config.num_heads
        num_heads_small = model_config.num_heads_small 

        self.model_config = model_config 
        self.ds_config = ds_config 

        self.device = device            
        self.num_layers = num_layers 
        self.layer_type = layer_type 
        self.heads = []
        self.num_heads = num_heads 

        for l in range(self.num_layers):
            dim_fac_last = num_heads[l-1]
            dim_fac_curr = num_heads[l]

            if layer_type == 'multi_hop':
                dim_fac_last = dim_fac_curr = 1 

            if l == self.num_layers - 1 == 0:         
                in_f, out_f = input_dim, output_dim
            elif l == 0:
                in_f, out_f = input_dim, hidden_dim
            elif l == self.num_layers - 1:
                in_f, out_f = dim_fac_last * hidden_dim, output_dim
            else:
                in_f, out_f = dim_fac_last * hidden_dim, hidden_dim

            do_concat = (l != self.num_layers - 1)  
            if layer_type == 'normal':
                layer = tgnn.GATConv(in_f, out_f, heads=num_heads[l], dropout=dropout, concat=do_concat, add_self_loops=True)
            elif layer_type == 'multi_hop':
                layer = HigherOrderGATHead(in_f, out_f, num_heads[l], do_concat, model_config)
            else:
                raise ValueError()
            self.heads.append(layer)

        self.heads = nn.ModuleList(self.heads)
        self.act_prelim = nn.ELU()
        self.act_final = nn.LogSoftmax(dim=-1)

        self.reset_parameters() 


    def forward(self, x, edge_index):
        if self.layer_type == 'normal':
            edge_index = len(self.heads) * [edge_index]
        
        for n, (head, ei) in enumerate(zip(self.heads, edge_index)):
            x = head(x, ei)
            if n != len(self.heads) - 1:
                x = self.act_prelim(x)
        
        x = self.act_final(x)

        return x


    def reset_parameters(self):
        for head in self.heads:
            head.reset_parameters()


class HigherOrderGATHead(nn.Module):
    def __init__(self, in_features, out_features, num_heads, do_concat, model_config, agg_type='sum'):
        super().__init__()

        self.num_heads_small = model_config.num_heads_small 
        self.num_hops = model_config.K_hops
        self.num_heads = num_heads

        self.heads = []
        for _ in range(self.num_heads):
            self.heads.append(HigherOrderGATLayer(
                                                    in_features, 
                                                    out_features, 
                                                    num_heads,
                                                    do_concat, 
                                                    model_config,
                                                    agg_type
            ))
        self.heads = nn.ModuleList(self.heads)
        self.concat = do_concat

        self.agg_type = model_config.agg_func  


    def forward(self, x, edge_index_list):
        pred_head = None
 
        for high_order_layer, edge_indexh in zip(self.heads, edge_index_list):
            pred = high_order_layer(x, edge_indexh).unsqueeze(-1)
            if pred_head is None:
                pred_head = pred
            else:
                pred_head = torch.concat((pred_head, pred), dim=-1)
        
        if self.agg_type == 'max':
            pred_head = torch.max(pred_head, dim=-1, keepdim=False).values
        elif self.agg_type == 'mean':
            pred_head = torch.mean(pred_head, dim=-1, keepdim=False)
        elif self.agg_type == 'sum':
            pred_head = torch.sum(pred_head, dim=-1, keepdim=False)
        else:
            raise ValueError() 

        return pred_head


    def reset_parameters(self):
        for head in self.heads:
            head.reset_parameters()


class TransConv(nn.Module):
    """
        Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    
    def __init__(self, in_features, out_features, opt, concat=True, edge_weights=None):
        super(TransConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = opt['leaky_relu_slope']
        self.concat = concat
        self.opt = opt
        self.h = int(opt['heads'])
        self.edge_weights = edge_weights 

        
        self.attention_dim = out_features
        if self.attention_dim % self.h != 0:
            self.h = 1 

        self.d_k = self.attention_dim // self.h

        self.Q = nn.Linear(in_features, self.attention_dim)
        self.init_weights(self.Q)

        self.V = nn.Linear(in_features, self.attention_dim)
        self.init_weights(self.V)

        self.K = nn.Linear(in_features, self.attention_dim)
        self.init_weights(self.K)

        self.activation = nn.Sigmoid()  # nn.LeakyReLU(self.alpha)

        self.Wout = nn.Linear(self.d_k, in_features)
        self.init_weights(self.Wout)

    def reset_parameters(self):
        self.Q.reset_parameters() 
        self.V.reset_parameters() 
        self.K.reset_parameters() 

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.constant_(m.weight, 1e-5)

    def forward(self, x, edge):
        """
        x might be [features, augmentation, positional encoding, labels]
        """
        if torch_sparse is None:
            raise ImportError("torch_sparse is required for TransConv head_type='transformer'.")

        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)

        # perform linear operation and split into h heads

        k = k.view(-1, self.h, self.d_k)
        q = q.view(-1, self.h, self.d_k)
        v = v.view(-1, self.h, self.d_k)

        # transpose to get dimensions [n_nodes, attention_dim, n_heads]

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        
        src = q[edge[0, :], :, :]
        dst_k = k[edge[1, :], :, :]

        prods = torch.sum(src * dst_k, dim=1) / np.sqrt(self.d_k)

        attention = softmax(prods, edge[0])

        concat_tensor = None 
        for h in range(self.h):
            vh = v[:,:,h]
            attnh = attention[:, h]
            pred = torch_sparse.spmm(edge, attnh, vh.shape[0], vh.shape[0], vh)

            if concat_tensor is None:
                concat_tensor = pred
            else:
                concat_tensor = torch.concat((concat_tensor, pred), dim=-1)
            
        return concat_tensor 

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


def combine_tensor_list(tlist: List[torch.tensor]):
  all_tensor = None 
  for n, tensor in enumerate(tlist):
    tensor = tensor.squeeze()
    if all_tensor is None:
      all_tensor = tensor 
    else:
      all_tensor = torch.cat((all_tensor, tensor), dim=-1)

  return all_tensor  


class HigherOrderGATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads, do_concat, model_config, agg_type: str='sum'):     # for now passing in number of heads for transformer attention 
        super().__init__()
        num_hops = model_config.K_hops 
        num_heads_small = model_config.num_heads_small 
        head_type = model_config.head_type
        dropout =  model_config.drop_out  

        self.beta_mul = model_config.beta_mul 
        self.agg_type = agg_type 

        self.head_type = model_config.head_type 
        self.attn_heads = [] 

        for k in range(num_hops):
            if self.head_type == 'gat':
                self.attn_heads.append(tgnn.GATConv( 
                                                in_features, 
                                                out_features, 
                                                dropout=dropout, 
                                                heads=num_heads_small,                             
                                                add_self_loops=True,   
                                                concat=False 
                ))
            elif self.head_type == 'transformer':
                leaky_slope = model_config.leaky_relu_slope 
                self.attn_heads.append(TransConv(                                      # SpGraphTransAttentionLayer will not work with GAT 
                                                                    in_features, 
                                                                    out_features, 
                                                                    model_config                      
                ))
            else:
                raise ValueError('Invalid Head Type')

        self.attn_heads = nn.ModuleList(self.attn_heads)


    def forward(self, features, edge_index_list): 
        dist_features = None 
        for k, (headk, edge_indexk) in enumerate(zip(self.attn_heads, edge_index_list)):  
            betak = 1 if k == 0 else self.beta_mul * 1/(k+1)

            pred = betak * headk(features, edge_indexk)
            pred = pred.unsqueeze(dim=-1)
            if dist_features is None:
                dist_features = pred
            else:
                dist_features = torch.concat((dist_features, pred), dim=-1)

        # potential problem area 
        if self.agg_type == 'sum':
            agg = tgnn.aggr.SumAggregation() 
            dist_features = agg(dist_features, dim=-1).squeeze(-1)
        elif self.agg_type == 'identity':
            dist_features = dist_features
        elif self.ret_alpha:
            dist_features = combine_tensor_list(dist_features)
        else:
            raise ValueError('Invalid Aggregation Argument') 

        return dist_features


    def reset_parameters(self):
        for head in self.attn_heads:
            head.reset_parameters()
