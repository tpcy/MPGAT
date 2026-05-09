from __future__ import division
from __future__ import print_function

import argparse
import time
import sys
import torch
import torch.nn.functional as F

import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.logging import init_wandb, log
from GATconv import GATConv
import torch_geometric.utils
# from utils3 import read_planetoid_data
from Planetoid import Planetoid
from share import kk

threshold = -0.1

class GAT(torch.nn.Module):
    def __init__(self, threshold, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, threshold, heads, dropout=0.6)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(hidden_channels * heads, out_channels, threshold, heads=1,
                             concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index, key = 0))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index, key = 1)
        return x

def add_two_hop_neighbors(edge_index):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_nodes = torch.max(edge_index) + 1
    # 构建邻接矩阵
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    adj_matrix[edge_index[0], edge_index[1]] = True
    adj_matrix = adj_matrix.to(device)
    # 计算邻接矩阵的平方，得到二跳邻居信息
    two_hop_adj_matrix = torch.mm(adj_matrix, adj_matrix)
    two_hop_edge_index = torch.nonzero(two_hop_adj_matrix).t()
    two_hop_edge_index = two_hop_edge_index.to(device)
    # 去除重复边
    new_edge_index = torch.cat([edge_index, two_hop_edge_index], dim=1)
    new_edge_index = torch_geometric.utils.coalesce(new_edge_index)
    return new_edge_index


def add_three_hop_neighbors(edge_index):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_nodes = torch.max(edge_index) + 1
    # 构建邻接矩阵
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    adj_matrix[edge_index[0], edge_index[1]] = True
    adj_matrix = adj_matrix.to(device)

    # 计算一跳邻接矩阵（原始边）
    one_hop_adj_matrix = adj_matrix
    one_hop_edge_index = torch.nonzero(one_hop_adj_matrix).t()
    one_hop_edge_index = one_hop_edge_index.to(device)

    # 计算二跳邻接矩阵（邻接矩阵的平方）
    two_hop_adj_matrix = torch.mm(adj_matrix, adj_matrix)
    two_hop_edge_index = torch.nonzero(two_hop_adj_matrix).t()
    two_hop_edge_index = two_hop_edge_index.to(device)

    # 计算三跳邻接矩阵（邻接矩阵的立方）
    three_hop_adj_matrix = torch.mm(two_hop_adj_matrix, adj_matrix)
    three_hop_edge_index = torch.nonzero(three_hop_adj_matrix).t()
    three_hop_edge_index = three_hop_edge_index.to(device)

    # 合并一跳、二跳和三跳的边信息
    new_edge_index = torch.cat([one_hop_edge_index, two_hop_edge_index, three_hop_edge_index], dim=1)

    # 去除重复边，合并相同的边
    new_edge_index = torch_geometric.utils.coalesce(new_edge_index)

    return new_edge_index


def append_lines_to_file(file_path, lines):
    with open(file_path, 'a', encoding='utf-8') as file:
        for line in lines:
            file.write(line + ' ')

def main(num_runs):
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--hidden_channels', type=int, default=8)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--wandb', action='store_true', help='Track experiment')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    global threshold
    # threshold += 0.01
    # if threshold > 0.2:
    #     sys.exit()

    kk[0] += 1
    if kk[0] > 100:
        sys.exit()

    for run in range(num_runs):
        init_wandb(name=f'GAT-{args.dataset}', heads=args.heads, epochs=args.epochs,
                   hidden_channels=args.hidden_channels, lr=args.lr, device=device)

        # 假设数据集已经下载到本地的固定目录
        local_dataset_path = './data/Planetoid/'  # 这里修改为你本地数据集的实际存储路径
        dataset = Planetoid(local_dataset_path, args.dataset, transform=T.NormalizeFeatures())
        data = dataset[0].to(device)



        # number = 1
        data.edge_index = add_two_hop_neighbors(data.edge_index)
        number = 2
        # data.edge_index = add_three_hop_neighbors(data.edge_index)
        # number = 3


        model = GAT(threshold, dataset.num_features, args.hidden_channels, dataset.num_classes,
                    args.heads).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)



        if torch.cuda.is_available():
            model = model.to(device)

        def train(epoch):
            t = time.time()
            model.train()
            optimizer.zero_grad()

            tt = time.time()
            output = model(data.x, data.edge_index)
            print('time: {:.4f}s'.format(time.time() - tt))

            loss_train = F.cross_entropy(output[data.train_mask], data.y[data.train_mask])
            loss_train.backward()
            optimizer.step()

            model.eval()
            pred = model(data.x, data.edge_index).argmax(dim=-1)
            accs = []
            for mask in [data.train_mask, data.val_mask, data.test_mask]:
                accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))

            acc_train, acc_val, acc_test = accs

            loss_val = F.cross_entropy(output[data.val_mask], data.y[data.val_mask])
            print('Epoch: {:04d}'.format(epoch+1),
                  'loss_train: {:.4f}'.format(loss_train.data.item()),
                  'acc_train: {:.4f}'.format(acc_train),
                  'loss_val: {:.4f}'.format(loss_val.data.item()),
                  'acc_val: {:.4f}'.format(acc_val),
                  'test_acc: {:.4f}'.format(acc_test),
                  'time: {:.4f}s'.format(time.time() - t))
            return acc_train, acc_val, acc_test


        # Train model
        t_total = time.time()
        best_test_acc = 0
        best_val_acc = 0
        for epoch in range(args.epochs):
            acc_train, acc_val, acc_test = train(epoch)
            if acc_val > best_val_acc:
                best_val_acc = acc_val
                best_test_acc = acc_test

        if run == 0:
            final_lines_to_append = [
                # "{:.4f}\n".format(threshold)
                "{:.1f}\n".format(kk[0])
            ]
            append_lines_to_file('example9.txt', final_lines_to_append)

        lines_to_append = [
            "数据集= {}".format(args.dataset),
            "跳数= {}".format(number),
            "轮数= {}".format(run + 1),
            "accuracy= {:.4f}\n".format(best_test_acc)
        ]
        append_lines_to_file('example9.txt', lines_to_append)
        acc_values.append(best_test_acc)
        if run == num_runs - 1:
            final_lines_to_append = [
                "final_average_accuracy= {:.5f}\n\n".format(sum(acc_values)/len(acc_values))
            ]
            append_lines_to_file('example9.txt', final_lines_to_append)
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))



if __name__ == "__main__":
    num_runs = 10
    for _ in range(100):
        acc_values = []
        main(num_runs)

