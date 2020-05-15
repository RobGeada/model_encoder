import numpy as np
import torch
import torch.nn as nn


class ModelEncoder(nn.Module):
    def __init__(self, input_dim, adjacency, operation_set, output_function, device):
        super().__init__()
        # initialize statistics
        self.input_dim = np.array(input_dim)
        self.adjacency = adjacency
        self.operation_set = operation_set
        self.cells, self.nodes, _, self.ops = adjacency.shape
        self.device = device

        # initialize dictionaries for model operations and tensor dimensionalities
        self.dims = {(i,j): None for i in range(self.cells) for j in range(self.nodes)}
        self.dims[0, 0] = self.input_dim
        self.ops = nn.ModuleDict()

        # build model from adjacency
        ops = np.where(self.adjacency==1)
        ops = list(zip(*ops))
        for (c, na, nb, o) in ops:
            key = self.encode_key(c, na, nb)
            in_dim = self.dims[c, na]

            # check to see if this node is ever actually reached
            if in_dim is not None:
                # add operation between node_a and node_b in cell c
                if key not in self.ops:
                    self.ops[key] = nn.ModuleList()
                self.ops[key].append(self.operation_set[o].function(c=int(in_dim[1])))

                # assign the dimensionality of node_b
                # if inbound edges carry different dimensions, raise an error
                out_dim = in_dim * self.operation_set[o].mod
                if self.dims[c, nb] is not None and not np.allclose(self.dims[c,nb], out_dim):
                    raise ValueError('Dimension mismatch at cell {}, node {}->node {}, operation {}'.format(
                        c, na, nb, self.operation_set[o].name))
                else:
                    self.dims[c, nb] = out_dim
                    if nb == self.nodes-1 and c != self.cells-1:
                        self.dims[c+1, 0] = out_dim

        # record all the nodes that each node connects to
        self.node_targets = {(c, na):
                                 self.get_all_targets(c,na) for c in range(self.cells) for na in range(self.nodes)}

        # for a particular node in a cell, record its tensor dimensionality
        self.node_sizes = \
            {c: {n: self.dims[c,n ].astype(int).tolist() for n in range(self.nodes) if self.dims[(c,n)] is not None}
             for c in range(self.cells)}

        # add a normalizer to each node to prevent summation overflow
        self.node_norms = nn.ModuleDict()
        for c in range(self.cells):
            cell_norms = {}
            for n, size in self.node_sizes[c].items():
                cell_norms[str(n)] = nn.BatchNorm2d(size[1])
            self.node_norms[str(c)] = nn.ModuleDict(cell_norms)

        self.output_function = output_function(self.dims[self.cells-1, self.nodes-1])

        # put on cuda if desired
        self.to(self.device)

    # generate adjacency matrix from model structure dictionary
    @staticmethod
    def adj_from_dict(d, operation_set):
        cells = max(d.keys()) + 1
        nodes = max([n for key in d[0].keys() for n in key]) + 1
        n_ops = len(operation_set)
        op_reference = {op.name: i for i, op in enumerate(operation_set)}
        adj = np.zeros((cells, nodes, nodes, n_ops))
        for c, nodes in d.items():
            for (na, nb), ops in nodes.items():
                for op in ops:
                    if type(op) is int:
                        adj[c, na, nb, op] = 1
                    elif type(op) is str:
                        op_idx = op_reference[op]
                        adj[c, na, nb, op_idx] = 1
        return adj

    # convert edge coordinates for a string (necessary for nn.ModuleDict implementation)
    def encode_key(self, c, na, nb):
        return "{}_{}_{}".format(c, na, nb)

    # convert string back to edge coordinates (necessary for nn.ModuleDict implementation)
    def decode_key(self, k):
        return [int(x) for x in k.split("_")]

    # create zero tensor on same device that the model currently resides on
    def zeros(self, dim):
        return torch.zeros(dim, device=self.device)

    # for a particular node in the model, return the edge keys and the target nodes of all outbound edges
    def get_all_targets(self, c, na):
        targets = []
        for key in self.ops.keys():
            key_c, key_na, key_nb = self.decode_key(key)
            if key_c == c and key_na == na:
                targets.append([key, key_nb])
        return targets

    def forward(self, x):
        cell_in = x
        batch_size = x.shape[0]
        for c in range(self.cells):
            # initialize output values of each node, include cellular input as node 0 output
            node_outputs = {n: self.zeros([batch_size]+size[1:]) if n else cell_in
                            for n, size in self.node_sizes[c].items()}
            # clear cellular input
            del cell_in

            # propagate operations through cell
            origins = list(node_outputs.keys())
            for node_a in origins:
                # if node is cellular output, add node output to next cell input
                if node_a == self.nodes-1:
                    cell_in = node_outputs[node_a]

                # compute the output of a node to each target node
                node_input = self.node_norms[str(c)][str(node_a)](node_outputs[node_a])
                for op_key, target in self.node_targets[c, node_a]:
                    for op in self.ops[op_key]:
                        node_outputs[target] += op(node_input)

                # delete node output, now that all downstream connections have been computed
                del node_outputs[node_a]
            del node_outputs

        return self.output_function(cell_in)
