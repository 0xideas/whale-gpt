

import numpy as np
import pandas as pd
import random
import string
from sklearn.metrics.pairwise import manhattan_distances

class TreeNode():
    def __init__(self, val):
        self.name = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(16))
        self.children = []
        self.val = val
    def addChild(self, tree_node):
        assert tree_node.name != self.name
        self.children.append(tree_node)
    
    def str(self, indent):
        return(f"{self.val}\n" + "\n".join([(" " * indent) + child.str(indent=indent+4) for child in self.children]))
    
    def __str__(self):
        return(f"{self.val}\n" + "\n".join([child.str(indent=4) for child in self.children]))
    
    def __repr__(self):
        return(str(self))


def coda_distances(sequence, means, only_equal=False):
    sequence_ = np.array(sequence)
    distance = {}
    for coda, mean in means.items():
        if coda != -1 and len(sequence) >= len(mean):
            sequence_normalized = np.cumsum(sequence_/sequence_.sum())[:len(mean)]
            n_equal_one = np.sum(np.abs(sequence_normalized - 1.0) < 1e-10)
            if (n_equal_one <=1 and not only_equal) or n_equal_one == 1:
                distance[coda] = manhattan_distances(sequence_normalized.reshape(1, -1), mean.reshape(1, -1))[0][0]

    return(distance)
    

def get_coda(sequence, means, only_equal=False):
    distances = coda_distances(sequence, means, only_equal)
    sorted_ = sorted(list(distances.items()), key=lambda x: x[1])
    if len(sorted_):
        return(sorted_[0])
    else:
        return(None)
    
def get_candidates_sorted_filtered(sequence, means, threshold=0.1, only_equal=True):
    distances = coda_distances(sequence, means, only_equal)
    print(distances)
    sorted_ = sorted(list(distances.items()), key=lambda x: x[1])
    candidates = [(coda, distance) for coda, distance in sorted_ if distance <= threshold]
    return(candidates)


def expand_tree(tree, candidates, sequence, sequence_eval_index, means, coda_lengths, limit, threshold, only_equal):
    for candidate in candidates:
        sequence_remainder = sequence[coda_lengths[candidate[0]]:]
        print(f"{sequence_remainder = }")
        child = TreeNode(candidate)
        if len(sequence_remainder) > 1:
            print("sequence_remainder 2")
                # tree, sequence, sequence_eval_index, means, coda_lengths, limit=3, threshold=0.1, only_equal=True
            child = get_coda_tree(child, sequence_remainder, sequence_eval_index, means, coda_lengths, limit, threshold, only_equal)
            print(f"{child.children = }")
            tree.addChild(child)
        elif len(sequence_remainder) == 1:
            child.addChild(TreeNode((100, 1.0)))
        tree.addChild(child)
        
    return(tree)


def get_coda_tree(tree, sequence, sequence_eval_index, means, coda_lengths, limit=3, threshold=0.1, only_equal=True):
    print(f"{sequence[:sequence_eval_index] = }")
    candidates = get_candidates_sorted_filtered(sequence[:sequence_eval_index], means, threshold, only_equal)[:limit]
    print(f"{candidates = }")
    if len(candidates) > 0:
        tree = expand_tree(tree, candidates, sequence, sequence_eval_index, means, coda_lengths, limit, threshold, only_equal)
    else:
        if not 100 in [child.val[0] for child in tree.children]:
            candidates1 = get_candidates_sorted_filtered(sequence[1:sequence_eval_index+1], means, threshold, only_equal)
            if len(candidates1) > 0:
                child1 = expand_tree(TreeNode((100, 1.0)), candidates1, sequence, sequence_eval_index, means, coda_lengths, limit, threshold, only_equal)
                tree.addChild(child1)

        if sequence_eval_index > 1:
            child2 = get_coda_tree(tree, sequence, sequence_eval_index-1, means, coda_lengths, limit, threshold, only_equal)
    return(tree)

