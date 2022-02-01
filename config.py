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
    with open(yaml_path, 'r') as f:
        yaml_config = yaml.load(f, Loader=yaml.SafeLoader)
    return yaml_config


def copy_configs_to_folder(to_folder):
    configs_from_dir = os.path.join(get_project_path(), "configs")
    to_folder = os.path.join(to_folder, "configs")
    if not os.path.exists(to_folder):
        shutil.copytree(configs_from_dir, to_folder)
    else:
        logging.info("File exists:", to_folder)
        key = input("Output directory already exists! Overwrite the folder? (y/n).")
        if key == 'y':
            shutil.rmtree(to_folder)
            shutil.copytree(configs_from_dir, to_folder)
        else:
            logging.info("Please respecify the folder.")

            sys.exit(1)


class Config(object):
    """Object to hold the config requirements for an agent/game"""

    def __init__(self):
        self.seed = 1

    def load_configs(self, yaml_path, parser_config):
        # read configs from yaml path
        yaml_abs_path = os.path.join(get_project_path(), yaml_path)
        if not os.path.exists(yaml_abs_path):
            logging.error("yaml_abs_path : {} not exist".format(yaml_abs_path))

        with open(yaml_abs_path, 'r') as f:
            yaml_config = yaml.load(f, Loader=yaml.SafeLoader)

        # setup folder
        self.setup_folder(yaml_config, parser_config)

        print("Config:", yaml_config)
        return yaml_config

    def setup_folder(self, yaml_config, parser_config):
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
            copy_configs_to_folder(yaml_config["out_folder"])


if __name__ == '__main__':
    config = Config().load_configs(yaml_path="configs/default_dqn.yaml")
    print()
