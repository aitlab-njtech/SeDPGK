import argparse
import copy
import itertools
from pathlib import Path
from models.utils import get_training_config, check_device
from data.get_dataset import get_experiment_config
from utils.logger import output_results
from hypersearch import AutoML, raw_experiment
from collections import defaultdict, namedtuple


num_layers = [10, 6, 5, 7, 8, 9]
emb_dim = [64, 32, 16, 8]
feat_drop = [0.8, 0.5, 0.2]
attn_drop = [0.2, 0.5, 0.8]
lr = [0.001, 0.005, 0.01]
wd = [0.01, 0.001, 0.0005]

predefined_configs = {
    'ind': {
        'GCN': {
            'ant-1.7.csv': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'camel-1.2.csv': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'camel-1.6.csv': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'jedit-3.2.csv': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'log4j-1.0.csv': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'log4j-1.2.csv': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
             'lucene-2.0.csv': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
              'lucene-2.2.csv': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
              'lucene-2.4.csv': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
              'poi-1.5.csv': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
              'poi-2.0.csv': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
              'poi-2.5.csv': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'poi-3.0.csv': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'synapse-1.0.csv': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'synapse-1.1.csv': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'synapse-1.2.csv': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'velocity-1.4.csv': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'velocity-1.6.csv': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'xalan-2.5.csv': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'xalan-2.6.csv': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'xerces-1.2.csv': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'xerces-1.3.csv': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'ant-1.7': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'camel-1.2': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'camel-1.6': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'jedit-3.2': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'log4j-1.0': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'log4j-1.2': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
             'lucene-2.0': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
              'lucene-2.2': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
              'lucene-2.4': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
              'poi-1.5': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
              'poi-2.0': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
              'poi-2.5': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'poi-3.0': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'synapse-1.0': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'synapse-1.1': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'synapse-1.2': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'velocity-1.4': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'velocity-1.6': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'xalan-2.5': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'xalan-2.6': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'xerces-1.2': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'xerces-1.3': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
        },
      'GAT': {
            'ant-1.7': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'camel-1.2': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'camel-1.6': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'jedit-3.2': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'log4j-1.0': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'log4j-1.2': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
             'lucene-2.0': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
              'lucene-2.2': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
              'lucene-2.4': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
              'poi-1.5': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
              'poi-2.0': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
              'poi-2.5': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'poi-3.0': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'synapse-1.0': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'synapse-1.1': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'synapse-1.2': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'velocity-1.4': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'velocity-1.6': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'xalan-2.5': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'xalan-2.6': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'xerces-1.2': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'xerces-1.3': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
        },
      'APPNP': {
            'ant-1.7': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'camel-1.2': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'camel-1.6': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'jedit-3.2': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'log4j-1.0': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'log4j-1.2': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
             'lucene-2.0': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
              'lucene-2.2': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
              'lucene-2.4': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
              'poi-1.5': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
              'poi-2.0': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
              'poi-2.5': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'poi-3.0': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'synapse-1.0': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'synapse-1.1': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'synapse-1.2': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'velocity-1.4': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'velocity-1.6': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'xalan-2.5': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'xalan-2.6': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'xerces-1.2': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'xerces-1.3': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
        },
      'GCNII': {
            'ant-1.7': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'camel-1.2': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'camel-1.6': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'jedit-3.2': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'log4j-1.0': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'log4j-1.2': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
             'lucene-2.0': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
              'lucene-2.2': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
              'lucene-2.4': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
              'poi-1.5': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
              'poi-2.0': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
              'poi-2.5': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'poi-3.0': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'synapse-1.0': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'synapse-1.1': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'synapse-1.2': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'velocity-1.4': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'velocity-1.6': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'xalan-2.5': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'xalan-2.6': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'xerces-1.2': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'xerces-1.3': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
        },
      'GraphSAGE': {
            'ant-1.7': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'camel-1.2': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'camel-1.6': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'jedit-3.2': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'log4j-1.0': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'log4j-1.2': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
             'lucene-2.0': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
              'lucene-2.2': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
              'lucene-2.4': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
              'poi-1.5': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
              'poi-2.0': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
              'poi-2.5': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'poi-3.0': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'synapse-1.0': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'synapse-1.1': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'synapse-1.2': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'velocity-1.4': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'velocity-1.6': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'xalan-2.5': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'xalan-2.6': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'xerces-1.2': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'xerces-1.3': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
        },
      'MoNet': {
            'ant-1.7': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'camel-1.2': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'camel-1.6': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'jedit-3.2': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'log4j-1.0': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
            'log4j-1.2': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
             'lucene-2.0': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
              'lucene-2.2': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
              'lucene-2.4': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
              'poi-1.5': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
              'poi-2.0': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
              'poi-2.5': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'poi-3.0': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'synapse-1.0': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'synapse-1.1': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'synapse-1.2': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'velocity-1.4': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'velocity-1.6': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'xalan-2.5': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'xalan-2.6': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'xerces-1.2': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
               'xerces-1.3': {
                'num_layers': 8,
                'emb_dim': 32,
                'feat_drop': 0.8,
                'attn_drop': 0.2,
                'beta': 0,
                'lr': 0.001,
                'wd': 0.01
            },
        }
    },
        'tra': {
            'GCN': {
                'ant-1.7': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'camel-1.2': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'camel-1.6': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'jedit-3.2': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'log4j-1.0': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'log4j-1.2': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'lucene-2.0': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'lucene-2.2': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'lucene-2.4': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'poi-1.5': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'poi-2.0': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'poi-2.5': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'poi-3.0': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'synapse-1.0': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'synapse-1.1': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'synapse-1.2': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'velocity-1.4': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'velocity-1.6': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'xalan-2.5': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'xalan-2.6': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'xerces-1.2': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'xerces-1.3': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
            },
            'GAT': {
                'ant-1.7': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'camel-1.2': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'camel-1.6': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'jedit-3.2': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'log4j-1.0': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'log4j-1.2': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'lucene-2.0': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'lucene-2.2': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'lucene-2.4': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'poi-1.5': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'poi-2.0': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'poi-2.5': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'poi-3.0': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'synapse-1.0': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'synapse-1.1': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'synapse-1.2': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'velocity-1.4': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'velocity-1.6': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'xalan-2.5': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'xalan-2.6': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'xerces-1.2': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'xerces-1.3': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
            },
            'APPNP': {
                'ant-1.7': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'camel-1.2': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'camel-1.6': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'jedit-3.2': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'log4j-1.0': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'log4j-1.2': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'lucene-2.0': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'lucene-2.2': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'lucene-2.4': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'poi-1.5': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'poi-2.0': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'poi-2.5': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'poi-3.0': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'synapse-1.0': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'synapse-1.1': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'synapse-1.2': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'velocity-1.4': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'velocity-1.6': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'xalan-2.5': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'xalan-2.6': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'xerces-1.2': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'xerces-1.3': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
            },
            'GCNII': {
                'ant-1.7': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'camel-1.2': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'camel-1.6': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'jedit-3.2': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'log4j-1.0': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'log4j-1.2': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'lucene-2.0': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'lucene-2.2': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'lucene-2.4': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'poi-1.5': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'poi-2.0': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'poi-2.5': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'poi-3.0': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'synapse-1.0': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'synapse-1.1': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'synapse-1.2': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'velocity-1.4': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'velocity-1.6': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'xalan-2.5': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'xalan-2.6': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'xerces-1.2': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'xerces-1.3': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
            },
            'GraphSAGE': {
                'ant-1.7': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'camel-1.2': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'camel-1.6': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'jedit-3.2': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'log4j-1.0': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'log4j-1.2': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'lucene-2.0': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'lucene-2.2': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'lucene-2.4': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'poi-1.5': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'poi-2.0': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'poi-2.5': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'poi-3.0': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'synapse-1.0': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'synapse-1.1': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'synapse-1.2': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'velocity-1.4': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'velocity-1.6': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'xalan-2.5': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'xalan-2.6': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'xerces-1.2': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'xerces-1.3': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
            },
            'MoNet': {
                'ant-1.7': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'camel-1.2': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'camel-1.6': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'jedit-3.2': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'log4j-1.0': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'log4j-1.2': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'lucene-2.0': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'lucene-2.2': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'lucene-2.4': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'poi-1.5': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'poi-2.0': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'poi-2.5': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'poi-3.0': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'synapse-1.0': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'synapse-1.1': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'synapse-1.2': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'velocity-1.4': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'velocity-1.6': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'xalan-2.5': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'xalan-2.6': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'xerces-1.2': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
                },
                'xerces-1.3': {
                    'num_layers': 8,
                    'emb_dim': 32,
                    'feat_drop': 0.8,
                    'attn_drop': 0.2,
                    'beta': 0,
                    'lr': 0.001,
                    'wd': 0.01
            },
        }
    }
}



def arg_parse(parser):
    parser.add_argument('--dataset', type=str, default='lucene-2.0', help='Dataset')
    parser.add_argument('--teacher', type=str, default='MoNet', help='Teacher Model')
    parser.add_argument('--student', type=str, default='PLP', help='Student Model')
    parser.add_argument('--distill', action='store_false', default=True, help='Distill or not')#?????????????????????????????????????????????True
    parser.add_argument('--device', type=int, default=0, help='CUDA Device')
    parser.add_argument('--ptype', type=str, default='ind', help='plp type: ind(inductive); tra(transductive/onehot)')
    parser.add_argument('--labelrate', type=int, default=30, help='label rate')
    parser.add_argument('--mlp_layers', type=int, default=2, help='MLP layer, 0 means not add feature mlp/lr')#2
    parser.add_argument('--grad', type=int, default=1, help='output grad or not')
    parser.add_argument('--automl', action='store_true', default=False, help='Automl or not')#??????????????????
    parser.add_argument('--ntrials', type=int, default=10, help='Number of trials')
    parser.add_argument('--njobs', type=int, default=10, help='Number of jobs')
    return parser.parse_args()


def set_configs(configs):
    configs = dict(configs, **predefined_configs[args.ptype][args.teacher][args.dataset])
    training_configs_path = Path.cwd().joinpath('models', 'train.conf.yaml')#?????????????????????????????????
    model_name = configs['student'] if configs['distill'] else configs['teacher']#??????configs['distill']???????????????????????????????????????????????????
    training_configs = get_training_config(training_configs_path, model_name)#????????????????????????
    configs = dict(configs, **training_configs)
    configs['device'] = check_device(configs)
    data_configs_path = Path.cwd().joinpath('data', 'dataset.conf.yaml')#?????????????????????????????????
    configs['division_seed'] = get_experiment_config(data_configs_path)['seed']
    return configs


def func_search(trial):
    return {
        "num_layers": trial.suggest_int("num_layers", 5, 10),
        "emb_dim": trial.suggest_categorical("emb_dim", [64, 32, 16, 8]),
        "feat_drop": trial.suggest_categorical("feat_drop", [0.8, 0.5, 0.2]),
        "attn_drop": trial.suggest_categorical("attn_drop", [0.8, 0.5, 0.2]),
        "learning_rate": trial.suggest_categorical("learning_rate", [1e-3, 5e-3, 1e-2]),
        "weight_decay": trial.suggest_categorical("weight_decay", [5e-4, 1e-3, 1e-2]),
    }


def gen_variants(**items):
    Variant = namedtuple("Variant", items.keys())#namedtuple??????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    return itertools.starmap(Variant, itertools.product(*items.values()))


def variant_args_generator(args, variants):
    """Form variants as group with size of num_workers"""
    for variant in variants:
        args.dataset, args.model, args.seed = variant
        yield copy.deepcopy(args)


if __name__ == '__main__':
    # load_configs
    args = arg_parse(argparse.ArgumentParser())#????????????????????????arg_parse??????
    configs = set_configs(args.__dict__)#??????????????????
    # model_train
    variants = list(gen_variants(dataset=[configs['dataset']],
                                 model=[configs['model_name']],
                                 seed=[configs['seed']]))          #????????????????????????????????????????????????????????????????????????
    results_dict = defaultdict(list)#??????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    for variant in variants:
        if args.automl:#????????????FALSE
            tool = AutoML(kwargs=configs, func_search=func_search)
            results, preds, labels, output = tool.run()
        else:
            results, preds, labels, output = raw_experiment(configs)
        results_dict[variant[:]] = [results]
        tablefmt = configs["tablefmt"] if "tablefmt" in configs else "github"
        print("\nFinal results:\n")

        output_results(results_dict, tablefmt)

    # # save outputs
    #save_output(output_dir, preds, labels, output, acc_test, same_predict, G, idx_train, adj, configs)
