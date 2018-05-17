import os
import io
import logging
import numpy as np
import configparser
import argparse
class Params(object):
    def __init__(self, index_dir = 'E:/qiuchi/clueweb12/index',dataset_name = "2013", base_model = 'GQLM', include_interaction = False, expansion_type = 'quantum', window_size = 5, max_dependency_count = 3,
    original_expansion_ratio = 0.5, current_history_ratio = 0.8,initial_list_size = 100, reranking_list_size = 20, top_term_num = 50,
    expanded_vocabulary_size = 20, dirichlet_mu = 2500):
        self.index_dir = index_dir
        self.dataset_name = dataset_name
        self.base_model = base_model
        self.include_interaction = include_interaction
        self.expansion_type = expansion_type
        self.window_size = window_size
        self.max_dependency_count = max_dependency_count
        self.original_expansion_ratio = original_expansion_ratio
        self.current_history_ratio = current_history_ratio
        self.initial_list_size = initial_list_size
        self.reranking_list_size = reranking_list_size
        self.top_term_num = top_term_num
        self.expanded_vocabulary_size = expanded_vocabulary_size
        self.dirichlet_mu = dirichlet_mu


    def parse_config(self, config_file_path):
        config = configparser.ConfigParser()
        config.read(config_file_path)
        config_common = config['COMMON']

        if 'index_dir' in config_common:
            self.index_dir = config_common['index_dir']

        if 'dataset_name' in config_common:
            self.dataset_name = config_common['dataset_name']

        if 'base_model' in config_common:
            self.base_model = config_common['base_model']

        if 'include_interaction' in config_common:
            self.include_interaction = bool(config_common['include_interaction'])

        if 'expansion_type' in config_common:
            self.expansion_type = config_common['expansion_type']

        if 'window_size' in config_common:
            self.window_size = int(config_common['window_size'])

        if 'max_dependency_count' in config_common:
            self.max_dependency_count = int(config_common['max_dependency_count'])

        if 'original_expansion_ratio' in config_common:
            self.original_expansion_ratio = float(config_common['original_expansion_ratio'])

        if 'current_history_ratio' in config_common:
            self.current_history_ratio = float(config_common['current_history_ratio'])

        if 'initial_list_size' in config_common:
            self.initial_list_size = int(config_common['initial_list_size'])

        if 'reranking_list_size' in config_common:
            self.reranking_list_size = int(config_common['reranking_list_size'])

        if 'top_term_num' in config_common:
            self.top_term_num = int(config_common['top_term_num'])

        if 'expanded_vocabulary_size' in config_common:
            self.expanded_vocabulary_size = int(config_common['expanded_vocabulary_size'])

        if 'dirichlet_mu' in config_common:
            self.dirichlet_mu = int(config_common['dirichlet_mu'])

    def export_to_config(self, config_file_path):
        config = configparser.ConfigParser()
        config['COMMON'] = {}
        config_common = config['COMMON']
        config_common['index_dir'] = self.index_dir
        config_common['dataset_name'] = self.dataset_name
        config_common['base_model'] = self.base_model
        config_common['include_interaction'] = str(self.include_interaction)
        config_common['expansion_type'] = self.expansion_type
        config_common['window_size'] = str(self.window_size)
        config_common['max_dependency_count'] = str(self.max_dependency_count)
        config_common['original_expansion_ratio'] = str(self.original_expansion_ratio)
        config_common['current_history_ratio'] = str(self.current_history_ratio)
        config_common['initial_list_size'] = str(self.initial_list_size)
        config_common['reranking_list_size'] = str(self.reranking_list_size)
        config_common['top_term_num'] = str(self.top_term_num)
        config_common['expanded_vocabulary_size'] = str(self.expanded_vocabulary_size)
        config_common['dirichlet_mu'] = str(self.dirichlet_mu)

        with open(config_file_path, 'w') as configfile:
            config.write(configfile)

    def parseArgs(self):
        #required arguments:
        parser = argparse.ArgumentParser(description='running the complex embedding network')
        parser.add_argument('-config', action = 'store', dest = 'config_file_path', help = 'The configuration file path.')
        args = parser.parse_args()
        self.parse_config(args.config_file_path)

def main():
    params = Params()
    config_file_path = 'config/config.ini'
    params.export_to_config(config_file_path)
if  __name__ == '__main__':
    main()
