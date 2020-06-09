from data_reader import DataReader
import argparse


def _parse_args():
    parser = argparse.ArgumentParser(description='driver.py')
    parser.add_argument('--d', type=str, default='mnist',
                        help='Pass data-set')
    parser.add_argument('--spl', type=float, default=0.1,
                        help='Provide train test split | fraction of data used for training')
    parser.add_argument('--b', type=int, default=32,
                        help='Batch Size')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Get Arguments
    # -------------------------------------------------
    args = _parse_args()
    print(args)

    # Get Data : Train Data Loader, Test Data Loader
    # -------------------------------------------------
    print("--- Fetching Data --- ")
    data_set = args.d
    batch_size = args.b
    split = args.spl

    data_reader = DataReader(batch_size=batch_size,
                             data_set=data_set,
                             download=True,
                             dev_set=True,
                             split=split)

    train_loader = data_reader.train_loader
    test_loader = data_reader.test_loader
