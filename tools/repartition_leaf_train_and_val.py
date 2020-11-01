"""
This splits the original Leaf's training data into the validation and smaller training sets
"""
import os, json, random, shutil

def load_leaf_train_json_paths(leaf_train_json_paths):
    """
    Load Leaf's training json file(s)
    """
    org_train_d = {}
    for leaf_train_json_path in leaf_train_json_paths:
        with open(leaf_train_json_path,'r') as f:
            d = json.load(f)
            if len(org_train_d) == 0:
                print("Loading {}".format(leaf_train_json_path))
                org_train_d = d
            else: # add a JSON element
                print("Adding the data in {}".format(leaf_train_json_path))
                org_train_d['num_samples'] += d['num_samples']
                org_train_d['users'] += d['users']
                for user in d['users']:
                    if user in org_train_d['user_data']:
                        org_train_d['user_data'][user] += d['user_data'][user]
                    else:
                        org_train_d['user_data'][user] = d['user_data'][user]

    return org_train_d


def split_train(leaf_train_json_paths, new_train_json_path, new_val_json_path, val_ratio):
    """
    Split Leaf's training set into the training and validation datasets
    """
    org_train_d = load_leaf_train_json_paths(leaf_train_json_paths)

    # create a validation dataset
    N = len(org_train_d['users'])
    if val_ratio > 0:
        val_idx = random.sample(range(0, N), int(val_ratio * N))
        b = {}
        b['num_samples'] = [org_train_d['num_samples'][n] for n in val_idx]
        b['users'] = [org_train_d['users'][n] for n in val_idx]
        b['user_data'] = dict(zip(b['users'], [org_train_d['user_data'][user] for user in b['users']]))

        os.makedirs(os.path.dirname(new_val_json_path), exist_ok=True)
        with open(new_val_json_path, 'w') as f:
            json.dump(b, f)

        print("Validation data: {}".format(new_val_json_path))
        print("#users={}".format(len(b['users'])))
        print("#samples={}".format(sum(b['num_samples'])))

    # remove val data from the original traning data
    if val_ratio > 0:
        train_idx = list(set(range(0, N)) - set(val_idx))
    else:
        train_idx = list(range(0, N))
    new_train_d = {}
    new_train_d['num_samples'] = [org_train_d['num_samples'][n] for n in train_idx]
    new_train_d['users'] = [org_train_d['users'][n] for n in train_idx]
    new_train_d['user_data'] = dict(zip(new_train_d['users'], [org_train_d['user_data'][user] for user in new_train_d['users']]))
    os.makedirs(os.path.dirname(new_train_json_path), exist_ok=True)
    with open(new_train_json_path, 'w') as f:
        json.dump(new_train_d, f)

    print("New training data file: {}".format(new_train_json_path))
    print("#users={}".format(len(new_train_d['users'])))
    print("#samples={}".format(sum(new_train_d['num_samples'])))


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-leaf_train_json_paths", nargs='+', default=["./data/train/all_data_0_0_keep_100_train_9.json"], help="Training JSON path")
    parser.add_argument("-new_train_json_path", default="./data/train/all_data_0_0_keep_100_train_9.json")
    parser.add_argument("-new_val_json_path", default="./data/val/all_data_0_0_keep_100_val_9.json")
    parser.add_argument("-val_ratio", type=float, default=0.1)
    parser.add_argument("-seed", type=int, default=123233)

    args = parser.parse_args()
    random.seed(args.seed)
    split_train(args.leaf_train_json_paths, args.new_train_json_path, args.new_val_json_path, args.val_ratio)

if __name__ == "__main__":
    main()
