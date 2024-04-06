from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles


def generate_scaffolds(dataset, log_every_n=1000):
    scaffolds = {}
    data_len = len(dataset)
    print(data_len)
    print("About to generate scaffolds")
    for idx, data in enumerate(dataset.predict_data):
        if idx % log_every_n == 0:
            print("Generating scaffold %d/%d" % (idx, data_len))
        mol = Chem.MolFromSmiles(data[0])

        if mol is None:
            continue

        scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [idx]
        else:
            scaffolds[scaffold].append(idx)

    # Sort from largest to smallest scaffold sets

    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    return scaffold_sets


def scaffold_split(dataset, valid_ratio, test_ratio, seed=None, log_every_n=1000):
    train_ratio = 1.0 - valid_ratio - test_ratio
    scaffold_sets = generate_scaffolds(dataset)
    train_cutoff = train_ratio * len(dataset)
    valid_cutoff = (train_ratio + valid_ratio) * len(dataset)
    train_idx = []
    valid_idx = []
    test_idx = []

    print("About to sort in scaffold sets")
    for scaffold_set in scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx += scaffold_set
            else:
                valid_idx += scaffold_set
        else:
            train_idx += scaffold_set
    return train_idx, valid_idx, test_idx
