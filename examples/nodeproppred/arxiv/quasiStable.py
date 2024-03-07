import torch_geometric
from ogb.nodeproppred import PygNodePropPredDataset
import numpy as np
import torch
import sys
sys.path.append("./examples/")
from QuasiStableColoring.quasiStableColoring import QuasiStableColoring


if __name__ == "__main__":
    dataset = PygNodePropPredDataset(name='ogbn-arxiv',
                                     transform=torch_geometric.transforms.ToSparseTensor(layout=torch.sparse_csr))
    data = dataset[0]
    print(data.num_nodes)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    data.to(device)
    qsc = QuasiStableColoring(data, device)
    qsc.q_color(n_colors=data.num_nodes * 0.01)