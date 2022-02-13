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


def copy_configs_to_folder(from_folder, to_folder):
    from_folder = os.path.join(from_folder, "configs")
    to_folder = os.path.join(to_folder, "configs")
    if not os.path.exists(to_folder):
        shutil.copytree(from_folder, to_folder)
    else:
        logging.info("File exists:{}".format(to_folder))
        key = input(
            "Output directory already exists! \nFrom {} to {}. \nOverwrite the folder? (y/n).".format(from_folder,
                                                                                                  to_folder))
        if key == 'y':
            shutil.rmtree(to_folder)
            shutil.copytree(from_folder, to_folder)
        else:
            logging.info("Please respecify the folder.")

            sys.exit(1)


def setup_folder(parser_args):
    """
    config all output folders and input folders, setup model folder, board folder, result_folder
    :param parser_args:
    :return:
    """
    # out_folder given by parser_args
    parser_args.out_model = os.path.join(parser_args.out_folder, "model")
    parser_args.out_board = os.path.join(parser_args.out_folder, "board_log")
    parser_args.out_result = os.path.join(parser_args.out_folder, "result_log")

    create_folders(
        [parser_args.out_folder, parser_args.out_model, parser_args.out_board, parser_args.out_result])

    if not parser_args.train:
        if parser_args.in_folder is not None and parser_args.in_folder != "":
            parser_args.in_model = os.path.join(parser_args.in_folder, "model")

            check_folders_exist([parser_args.in_folder, parser_args.in_model])


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_folder", type=str, default="test_folder")
    parser.add_argument("--in_folder", type=str, default=None)
    parser.add_argument("--in_model_index", type=int)
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--head", action="store_true", default=False)

    parser_args = parser.parse_args()

    # set out_folder path and in_folder_path
    if parser_args.train:
        assert parser_args.out_folder
        parser_args.out_folder = os.path.join(get_project_path(), "output", parser_args.out_folder)

    if not parser_args.train or parser_args.resume:
        assert parser_args.in_folder and parser_args.in_model_index
        parser_args.out_folder = os.path.join(get_project_path(), "output", parser_args.out_folder)
        parser_args.in_folder = os.path.join(get_project_path(), "output", parser_args.in_folder)

    # config dir
    if parser_args.train:
        # copy configs dir from /project_path/configs to /project_path/output/out_folder/configs
        copy_configs_to_folder(get_project_path(), parser_args.out_folder)
    else:
        # copy configs dir from /project_path/output/in_folder/configs to /project_path/output/out_folder/configs
        copy_configs_to_folder(parser_args.in_folder, parser_args.out_folder)

    setup_folder(parser_args)

    # load some yaml files
    parser_args.env_config = read_yaml(os.path.join(parser_args.out_folder, "configs"), "env.yaml")
    parser_args.env_config_ros = read_yaml(os.path.join(parser_args.out_folder, "configs"), "env_ros.yaml")
    parser_args.agents_config = read_yaml(os.path.join(parser_args.out_folder, "configs"), "agents.yaml")
    parser_args.training_config = read_yaml(os.path.join(parser_args.out_folder, "configs"), "training.yaml")

    print("\nYaml env_config config:", parser_args.env_config)
    print("\nYaml agents_config config:", parser_args.agents_config)
    print("\nYaml training config:", parser_args.training_config)
    print("\n==============================================================================================\n")
    return parser_args
