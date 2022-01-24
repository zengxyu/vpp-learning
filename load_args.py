import argparse

from configs.config import Config


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_folder", type=str, default="test_folder")
    parser.add_argument("--in_folder", type=str, default=None)
    parser.add_argument("--in_model_index", type=int)
    parser.add_argument("--train", action="store_true", default=True)
    parser.add_argument("--render", action="store_true", default=False)

    parser_config = parser.parse_args()
    if parser_config.train:
        assert parser_config.out_folder
    else:
        assert parser_config.in_folder and parser_config.in_model_index

    return parser_config


def load_dqn_args():
    parser_config = process_args()
    config = Config().load_configs(yaml_path="configs/training_default_dqn.yaml", parser_config=parser_config)
    return parser_config, config


def load_ac_args():
    parser_config = process_args()
    config = Config().load_configs(yaml_path="configs/training_default_ac.yaml", parser_config=parser_config)
    return parser_config, config
