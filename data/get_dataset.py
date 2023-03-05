import argparse
import yaml
from pathlib import Path
import numpy as np
from data.make_dataset import get_dataset, get_train_val_test_split


def get_experiment_config(config_path):
    with open(config_path, 'r') as conf:
        return yaml.load(conf, Loader=yaml.FullLoader)


def generate_data_path(dataset, dataset_source):
    if dataset_source == 'npz':
        return 'data/npz/' + dataset + '.npz'
    else:
        print(dataset_source)
        raise ValueError(f'The "dataset_source" must be set to "npz"')


def load_dataset_and_split(labelrate, dataset):
    _config = {
        'dataset_source': 'npz',
        'seed': 0,
        'train_config': {
            'split': {
                'train_examples_per_class': labelrate,  # 20
                'val_examples_per_class': 30
            },
            'standardize_graph': True
        }
    }
    _config['data_path'] = generate_data_path(dataset, _config['dataset_source'])#获取数据集路径
    #获取邻接矩阵、特征矩阵和标签
    adj, features, labels = get_dataset(dataset, _config['data_path'],
                                            _config['train_config']['standardize_graph'],
                                            _config['train_config']['split']['train_examples_per_class'],
                                            _config['train_config']['split']['val_examples_per_class'])
    random_state = np.random.RandomState(_config['seed'])#获得随机数生成器_config['seed']
    #划分训练集、验证集和测试集
    idx_train, idx_val, idx_test = get_train_val_test_split(random_state, labels,
                                                                **_config['train_config']['split'])
    return adj, features, labels, idx_train, idx_val, idx_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='load data', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-c',
                        '--config-file',
                        type=str,
                        default='dataset.conf.yaml',
                        help='Path to the YAML configuration file for the experiment.')
    args = parser.parse_args()
    adj, features, labels, idx_train, idx_val, idx_test = load_dataset_and_split(args.config_file)
