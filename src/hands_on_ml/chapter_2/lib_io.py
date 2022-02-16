import os
import re


def create_directories(paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)


def set_directories(config_data):
    cur_file_path = os.path.dirname(os.path.abspath(__file__))
    root_dir = ''.join(re.split('(hands-on-ml)', cur_file_path)[:2])
    work_dir = os.path.join(root_dir, config_data['base_path'])
    os.chdir(work_dir)
    create_directories(['resources',
                        config_data['output_path'],
                        config_data['data_path'],
                        os.path.join(config_data['data_path'], 'housing')])