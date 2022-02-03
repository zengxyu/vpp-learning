import argparse
import logging
import os.path
import shutil
import sys

import yaml

from utilities.util import get_project_path


def create_folders(folders):
    """create out folder"""
    for folder in folders:
        if not os.path.exists(folder):
            print("Create folder:{}", folder)
            os.makedirs(folder)


def check_folders_exist(folders):
    """check if folders exist"""
    for folder in folders:
        assert os.path.exists(folder), "Path to folder : {} not exist!".format(folder)


def get_log_level(name):
    log_level_mapping = {"NOTSET": logging.NOTSET, "DEBUG": logging.DEBUG, "INFO": logging.INFO,
                         "WARNING": logging.WARNING, "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL}
    return log_level_mapping.get(name)


def read_yaml(config_dir, config_name):
    yaml_path = os.path.join(config_dir, config_name)

    # read configs from yaml path
    if not os.path.exists(yaml_path):
        logging.error("yaml_abs_path : {} not exist".format(yaml_path))

    with open(yaml_path, 'r') as f:
        yaml_config = yaml.load(f, Loader=yaml.SafeLoader)
    return yaml_config


def copy_configs_to_folder(configs_from_dir, to_folder):
    to_folder = os.path.join(to_folder, "configs")
    if not os.path.exists(to_folder):
        shutil.copytree(configs_from_dir, to_folder)
    else:
        logging.info("File exists:", to_folder)
        key = input(
            "Output directory already exists! From {} to {}. Overwrite the folder? (y/n).".format(configs_from_dir,
                                                                                                  to_folder))
        if key == 'y':
            shutil.rmtree(to_folder)
            shutil.copytree(configs_from_dir, to_folder)
        else:
            logging.info("Please respecify the folder.")

            sys.exit(1)


def setup_folder(yaml_config, parser_config):
    """
    config all output folders and input folders
    :param parser_config:
    :param yaml_config:
    :return:
    """
    yaml_config["out_parent_folder"] = os.path.join(get_project_path(), yaml_config["out_parent_folder"])

    # out_folder given by parser_args
    yaml_config["out_folder"] = os.path.join(yaml_config["out_parent_folder"], parser_config.out_folder)
    yaml_config["out_model"] = os.path.join(yaml_config["out_folder"], yaml_config["out_model"])
    yaml_config["out_board"] = os.path.join(yaml_config["out_folder"], yaml_config["out_board"])
    yaml_config["out_result"] = os.path.join(yaml_config["out_folder"], yaml_config["out_result"])

    create_folders(
        [yaml_config["out_folder"], yaml_config["out_model"], yaml_config["out_board"],
         yaml_config["out_result"]])

    if parser_config.in_folder is not None and parser_config.in_folder != "":
        yaml_config["in_folder"] = os.path.join(yaml_config["out_parent_folder"], parser_config.in_folder)
        yaml_config["in_model"] = os.path.join(yaml_config["in_folder"], yaml_config["in_model"])

        check_folders_exist(
            [yaml_config["in_folder"], yaml_config["in_model"]])

    if parser_config.train:
        copy_configs_to_folder(parser_config.configs_dir, yaml_config["out_folder"])


def load_training_configs(parser_config):
    yaml_training_config = read_yaml(parser_config.configs_dir, "training_default_dqn.yaml")

    # setup folder
    setup_folder(yaml_training_config, parser_config)

    print("Yaml training config:", yaml_training_config)

    return yaml_training_config


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_folder", type=str, default="test_folder")
    parser.add_argument("--in_folder", type=str, default=None)
    parser.add_argument("--in_model_index", type=int)
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--configs_dir", default="")

    parser_config = parser.parse_args()
    if parser_config.train:
        assert parser_config.out_folder
    else:
        assert parser_config.in_folder and parser_config.in_model_index

    # config dir
    if parser_config.train:
        parser_config.configs_dir = os.path.join(get_project_path(), "configs")
    else:
        parser_config.configs_dir = os.path.join("output", parser_config.in_folder, "configs")

    return parser_config


def get_configs_dir():
    return parser_config.configs_dir


def load_dqn_args():
    return parser_config, training_configs


def load_ac_args():
    return parser_config, training_configs


parser_config = process_args()

training_configs = load_training_configs(parser_config)

env_config = read_yaml(get_configs_dir(), "env.yaml")

print(training_configs)
