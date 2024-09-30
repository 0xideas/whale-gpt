import random
import string

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import manhattan_distances, cosine_distances, euclidean_distances

rhythm = {
    "1+1+3": 5,
    "1+31": 4,
    "1+32": 4,
    "1-NOISE": -1,
    "10-NOISE": -1,
    "10R": 16,
    "10i": 17,
    "2+3": 7,
    "2-NOISE": -1,
    "3-NOISE": -1,
    "3D": 0,
    "3R": 1,
    "4-NOISE": -1,
    "4D": 2,
    "4R1": 3,
    "4R2": 3,
    "5-NOISE": -1,
    "5R1": 6,
    "5R2": 6,
    "5R3": 6,
    "6-NOISE": -1,
    "6R": 8,
    "6i": 9,
    "7-NOISE": -1,
    "7D1": 10,
    "7D2": 10,
    "7R": 10,
    "7i": 11,
    "8-NOISE": -1,
    "8D": 12,
    "8R": 12,
    "8i": 13,
    "9-NOISE": -1,
    "9R": 14,
    "9i": 15,
}


class TreeNode:
    def __init__(self, val):
        self.name = "".join(
            random.choice(string.ascii_letters + string.digits) for _ in range(16)
        )
        self.children = []
        self.val = val

    def addChild(self, tree_node):
        assert tree_node.name != self.name
        self.children.append(tree_node)

    def str(self, indent):
        return f"{self.val}\n" + "\n".join(
            [(" " * indent) + child.str(indent=indent + 4) for child in self.children]
        )

    def __str__(self):
        return f"{self.val}\n" + "\n".join(
            [child.str(indent=4) for child in self.children]
        )

    def __repr__(self):
        return str(self)

    def length(self, coda_lengths, cumsum, indent=0):
        newsum = coda_lengths[self.val[0]] + cumsum
        return f"{newsum}\n" + "\n".join(
            [
                (" " * indent) + child.length(coda_lengths, newsum, indent=indent + 4)
                for child in self.children
            ]
        )

    def get_best_path(self, first_call=True, path=[], score=0.0, extra_value=0.05):
        new_path = path + [(self.val[0], self.val[2], self.val[3])]
        if self.val[0] == 100:
            new_score = score + extra_value
        else:
            new_score = score + self.val[1]
        if len(self.children) == 0:
            if not first_call:
                return [(new_path, new_score)]
            else:
                return (new_path, new_score)
        else:
            children = [
                child.get_best_path(False, new_path, new_score, extra_value)
                for child in self.children
            ]
            children = [x for y in children for x in y]
            children = sorted(children, key=lambda x: x[1])
            if first_call:
                return (children[0][0][1:], children[0][1])
            else:
                return children


def coda_distances(sequence, means, only_equal=True):
    sequence_ = np.array(sequence)
    distance = {}
    for coda, mean in means.items():
        if coda != -1 and len(sequence) >= len(mean):
            sequence_normalized = np.cumsum(sequence_ / sequence_.sum())[: len(mean)]
            n_equal_one = np.sum(np.abs(sequence_normalized - 1.0) < 1e-10)
            if (n_equal_one <= 1 and not only_equal) or n_equal_one == 1:
                distance[coda] = manhattan_distances(
                    sequence_normalized.reshape(1, -1), mean.reshape(1, -1)
                )[0][0]

    return distance


def get_coda(sequence, means, only_equal=True):
    distances = coda_distances(sequence, means, only_equal)
    sorted_ = sorted(list(distances.items()), key=lambda x: x[1])
    if len(sorted_):
        return sorted_[0]
    else:
        return None


def get_candidates_sorted_filtered(sequence, means, threshold=0.1, only_equal=True):
    distances = coda_distances(sequence, means, only_equal)
    sorted_ = sorted(list(distances.items()), key=lambda x: x[1])
    candidates = [
        (coda, distance) for coda, distance in sorted_ if distance <= threshold
    ]
    return candidates


def expand_tree(
    tree,
    candidates,
    sequence,
    sequence_eval_index,
    sequence_start,
    means,
    coda_lengths,
    limit,
    threshold,
    only_equal,
):
    for candidate in candidates:
        sequence_length = min(len(sequence), coda_lengths[candidate[0]])
        sequence_end = sequence_start + sequence_length
        sequence_remainder = sequence[sequence_length:]
        child = TreeNode(
            (*candidate, sequence_start, sequence_end, sequence[:sequence_length])
        )
        if len(sequence_remainder) > 1:
            # tree, sequence, sequence_eval_index, means, coda_lengths, limit=3, threshold=0.1, only_equal=True
            child = get_coda_tree(
                tree=child,
                sequence=sequence_remainder,
                sequence_eval_index=sequence_eval_index,
                sequence_start=sequence_end,
                means=means,
                coda_lengths=coda_lengths,
                limit=limit,
                threshold=threshold,
                only_equal=only_equal,
            )
            tree.addChild(child)
        elif len(sequence_remainder) == 1:
            child.addChild(TreeNode((100, 1.0, sequence_end, sequence_end + 1)))
        tree.addChild(child)

    return tree


def get_coda_tree(
    tree,
    sequence,
    sequence_eval_index,
    sequence_start,
    means,
    coda_lengths,
    limit=3,
    threshold=0.1,
    only_equal=True,
):
    candidates = get_candidates_sorted_filtered(
        sequence[:sequence_eval_index], means, threshold, only_equal
    )[:limit]
    if len(candidates) > 0:
        tree = expand_tree(
            tree=tree,
            candidates=candidates,
            sequence=sequence,
            sequence_eval_index=sequence_eval_index,
            sequence_start=sequence_start,
            means=means,
            coda_lengths=coda_lengths,
            limit=limit,
            threshold=threshold,
            only_equal=only_equal,
        )
    else:
        if not 100 in [child.val[0] for child in tree.children]:
            candidates1 = get_candidates_sorted_filtered(
                sequence[1 : sequence_eval_index + 1], means, threshold, only_equal
            )
            if len(candidates1) > 0:

                child1 = expand_tree(
                    tree=TreeNode((100, 1.0, sequence_start, sequence_start + 1)),
                    candidates=candidates1,
                    sequence=sequence[1:],
                    sequence_eval_index=sequence_eval_index,
                    sequence_start=sequence_start + 1,
                    means=means,
                    coda_lengths=coda_lengths,
                    limit=limit,
                    threshold=threshold,
                    only_equal=only_equal,
                )
                tree.addChild(child1)

        if sequence_eval_index > 1:
            child2 = get_coda_tree(
                tree=tree,
                sequence=sequence,
                sequence_eval_index=sequence_eval_index - 1,
                sequence_start=sequence_start,
                means=means,
                coda_lengths=coda_lengths,
                limit=limit,
                threshold=threshold,
                only_equal=only_equal,
            )
    return tree


def standardize(dataframe, len):
    sums = np.sum(dataframe.values[:, :len], axis=1).reshape(-1, 1)
    denom = np.concatenate([sums for _ in range(len)], axis=1)
    cumsum = np.cumsum(dataframe.values[:, :len] / denom, axis=1)
    return np.concatenate([np.zeros((cumsum.shape[0], 1)), cumsum], axis=1)


def get_coda_data(path="data/DominicaCodas.csv"):
    codas = pd.read_csv(path)
    codas["CodaTypeConverted"] = [rhythm[v] for v in codas["CodaType"]]
    codas_groups = dict(
        tuple(codas.groupby("CodaTypeConverted")[[f"ICI{i+1}" for i in range(9)]])
    )
    lengths = {
        k: (np.min(np.sum(v.values != 0, 1)), np.max(np.sum(v.values != 0, 1)))
        for k, v in codas_groups.items()
    }
    standardised = {
        k: standardize(v, lengths[k][0]) for k, v in codas_groups.items() if k != -1
    }
    means = {k: v.mean(axis=0) for k, v in standardised.items()}
    means_trimmed = {k: v[1:] for k, v in means.items()}
    coda_lengths = {k: len(v) for k, v in means_trimmed.items()}

    return (means_trimmed, coda_lengths)


if __name__ == "__main__":
    dialogues = pd.read_csv("data/sperm-whale-dialogues.csv")
    means, coda_lengths = get_coda_data()
    results = {}
    for i in range(dialogues.shape[0]):
        sequence = np.array(
            list(dialogues[[f"ICI{i+1}" for i in range(28)]].values[i, :])
        )
        sequence = sequence[sequence > 0]
        tree = get_coda_tree(
            TreeNode((None, 0.0, 0, 0)),
            list(sequence),
            9,
            0,
            means,
            {k: len(v) for k, v in means.items()},
            limit=100,
            threshold=0.1
        )
        results[i] = (tree.get_best_path(extra_value=0.05), sequence)

    new_rows = []
    for i, ((path_tuples, score), sequence) in results.items():
        for id_, start, end in path_tuples:
            assert end <= len(sequence), f"{path_tuples = } - {sequence = }"
            delta = 9 - (end - start)
            assert (delta) >= 0, f"{path_tuples = } - {sequence = } - {delta = }"
            buffer = [0.0] * max(0, delta)
            tsToDelta = np.sum(sequence[:start])
            new_row = (
                list(dialogues.iloc[i, :][["REC", "nClicks", "Whale"]].values)
                + [dialogues.iloc[i, :]["TsTo"] + tsToDelta]
                + [i, id_, np.sum(sequence[start:end])]
                + list(sequence[start:end])
                + buffer
            )
            assert isinstance(new_row, list)
            new_rows = new_rows + [new_row]
    new_data = pd.DataFrame(
        data=new_rows,
        columns=["REC", "nClicks", "Whale", "TsTo", "Vocalization", "Coda", "Duration"]
        + [f"ICI{i+1}" for i in range(9)],
    )

    new_data["Coda"] = [int(v) if not pd.isnull(v) else -1 for v in new_data["Coda"]]
    new_data.to_csv("data/sperm-whale-dialogues-codas-manhattan.csv", index=False)
