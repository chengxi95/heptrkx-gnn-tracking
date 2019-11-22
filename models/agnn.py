"""
This module implements the PyTorch modules that define the sparse
message-passing graph neural networks for segment classification.
In particular, this implementation utilizes the pytorch_geometric
and supporting libraries:
https://github.com/rusty1s/pytorch_geometric
"""

# Externals
import torch
import torch.nn as nn
from torch_scatter import scatter_add
import logging
# Locals
from .utils import make_mlp

class EdgeNetwork(nn.Module):
    """
    A module which computes weights for edges of the graph.
    For each edge, it selects the associated nodes' features
    and applies some fully-connected network layers with a final
    sigmoid activation.
    """
    def __init__(self, input_dim, hidden_dim=8, hidden_activation='Tanh',
                 layer_norm=True):
        super(EdgeNetwork, self).__init__()
        self.network = make_mlp(input_dim*2,
                                [hidden_dim, hidden_dim, hidden_dim, 1],
                                hidden_activation=hidden_activation,
                                output_activation=None,
                                layer_norm=layer_norm)

    def forward(self, x, edge_index):
        # Select the features of the associated nodes
        start, end = edge_index
        x1, x2 = x[start], x[end]
        edge_inputs = torch.cat([x[start], x[end]], dim=1)
        return self.network(edge_inputs).squeeze(-1)

class NodeNetwork(nn.Module):
    """
    A module which computes new node features on the graph.
    For each node, it aggregates the neighbor node features
    (separately on the input and output side), and combines
    them with the node's previous features in a fully-connected
    network to compute the new features.
    """
    def __init__(self, input_dim, output_dim, hidden_activation='Tanh',
                 layer_norm=True):
        super(NodeNetwork, self).__init__()
        self.network = make_mlp(input_dim*3, [output_dim]*4,
                                hidden_activation=hidden_activation,
                                output_activation=hidden_activation,
                                layer_norm=layer_norm)

    def forward(self, x, e, edge_index):
        start, end = edge_index
        # Aggregate edge-weighted incoming/outgoing features
        mi = scatter_add(e[:, None] * x[start], end, dim=0, dim_size=x.shape[0])
        mo = scatter_add(e[:, None] * x[end], start, dim=0, dim_size=x.shape[0])
        node_inputs = torch.cat([mi, mo, x], dim=1)
        return self.network(node_inputs)

class GNNSegmentClassifier(nn.Module):
    """
    Segment classification graph neural network model.
    Consists of an input network, an edge network, and a node network.
    """
    def __init__(self, input_dim=3, hidden_dim=8, n_graph_iters=3,
                 hidden_activation='Tanh', layer_norm=True):
        super(GNNSegmentClassifier, self).__init__()
        self.n_graph_iters = n_graph_iters
        self.hidden_dim = hidden_dim
        # Setup the input network
        self.input_network = make_mlp(input_dim, [hidden_dim],
                                      output_activation=hidden_activation,
                                      layer_norm=layer_norm)
        # Setup the edge network
        self.edge_network = EdgeNetwork(hidden_dim, hidden_dim,
                                        hidden_activation, layer_norm=layer_norm)
        # Setup the node layers
        self.node_network = NodeNetwork(hidden_dim, hidden_dim,
                                        hidden_activation, layer_norm=layer_norm)

        # Set LSTM layer to combine all node information
        #self.combine_layer = nn.LSTM(hidden_dim, hidden_dim)
        
        # Setup the output layers
        #self.output_network = make_mlp(hidden_dim, [hidden_dim, hidden_dim, hidden_dim, 1], output_activation=hidden_activation, layer_norm=layer_norm)
        self.output_network = nn.Sequential(#nn.Linear(hidden_dim, hidden_dim),
                                            #nn.ReLU(),
                                            #nn.Linear(hidden_dim, hidden_dim),
                                            #nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, 3)
                                            )

    def forward(self, inputs):
        """Apply forward pass of the model"""

        # Apply input network to get hidden representation
        logging.debug(f'input x size: {inputs.x.shape}')
        x = self.input_network(inputs.x)
        logging.debug(f'x size after input network: {x.shape}')
        # Shortcut connect the inputs onto the hidden representation
        #x = torch.cat([x, inputs.x], dim=-1)

        # initalize global feature with random value
        #global_feature = torch.randn(self.hidden_dim, device=torch.device('cuda:0'))
        #logging.debug(f'global feature shape after initalization {global_feature.shape}')
        # Loop over iterations of edge and node networks
        for i in range(self.n_graph_iters):

            # Previous hidden state
            x0 = x

            # Apply edge network
            e = torch.sigmoid(self.edge_network(x, inputs.edge_index))
            logging.debug(f'shape after edge network: {e.shape}')

            # Apply node network
            x = self.node_network(x, e, inputs.edge_index)
            logging.debug(f'shape after node network: {x.shape}') 
            # Shortcut connect the inputs onto the hidden representation
            #x = torch.cat([x, inputs.x], dim=-1)

            # Residual connection
            x = x + x0

        # use LSTM to combine all node information into one
        #full_node, (global_node, cn) = self.combine_layer(x.view(x.shape[0],1,-1))
        #logging.debug(f'shape after LSTM: {global_node.shape}')
            
            #global_node = global_node[:, global_node.shape[-2]-1, :].squeeze()
            #logging.debug(f'shape after slice: {global_node.shape}')

            # feed global node and previous global feature to update global feature
            #logging.debug(f'shape of global feature; {global_feature.shape}')
            #logging.debug(f'shape of the cat tensor: {torch.cat([global_feature, global_node]).shape}')

        # Apply final edge network
        logging.debug(f'shape of x: {x.shape}')
        combine_node = torch.sum(x, dim=0)
        #return self.edge_network(x, inputs.edge_index)
        logging.debug(f'shape of sum tensor: {combine_node.shape}')
        output_node = self.output_network(combine_node.view(1, -1))
        logging.debug(f'shape of output: {output_node.shape}')
        return output_node
        #return self.output_network(x).squeeze(-1)
        #return self.output_network(torch.cat([global_feature, global_node]))
