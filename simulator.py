import torch
import torch.nn as nn
import torch_geometric as pyg

device = "cuda" if torch.cuda.is_available() else "cpu"

#temporary data container, naš dataset je preprocessed?
"""
class Graph:
    def __init__(self, nodes, edges, senders, receivers, globals=None):
        #nodes: [N, node_dim]
        #edges: [E, edge_dim]
        #senders, receivers: [E]
        #globals: [1, global_dim] or None
        
        self.nodes = nodes
        self.edges = edges
        self.senders = senders
        self.receivers = receivers
        self.globals = globals

    def to(self, device):
        self.nodes = self.nodes.to(device)
        self.edges = self.edges.to(device)
        self.senders = self.senders.to(device)
        self.receivers = self.receivers.to(device)
        if self.globals is not None:
            self.globals = self.globals.to(device)
        return self
"""

#WORK IN PROGRESS
#encoding,decoding,processing
class EPDBlock(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()

        self.edge_mlp = nn.Sequential(
            nn.Linear(2*node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )

    def forward(self, graph: Graph):
        #treba message passing,aggregation in update
        return graph

#main GNS class
class Simulator(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            EPDBlock(node_dim, edge_dim, hidden_dim) for _ in range(num_blocks)
        ])

    #one simulation step, sam poda naprej za zdj
    def forward(self, graph: Graph):
        for block in self.blocks:
            graph=block(graph)
        return graph

    def message(self):

    def update(self):


    #TODO treba bolj podrobno preverit kk se encoder/decoder processing dela za GPU
    #encoder
    def encoder_preprocessing(self):

    #decoder
    def decoder_postprocessing(self):

    #inverse decoder
    def inverse_decoder_postprocessing(self):