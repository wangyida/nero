import argparse

from train.trainer import Trainer
from utils.base_utils import load_cfg, load_config

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str)
"""
flags = parser.parse_args()
Trainer(load_cfg(flags.cfg)).run()
"""

args, extras = parser.parse_known_args()
Trainer(load_config(args.cfg, cli_args=extras)).run()
