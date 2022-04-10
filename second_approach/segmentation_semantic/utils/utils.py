
from json import load
import os
from dotenv import dotenv_values
import numpy as np
import re
import sys

def get_class_dict():
    return {
        'name' : ['mask', 'other'], 
        'color' : [[0,0,0],[255,255,255]]
    }

def get_data_dir(fold="train"):
    config = load_config()
    DATA_DIR = config["DATA_DIR"]

    dir = {
        "train": {
            "x": os.path.join(DATA_DIR, 'train'),
            "y": os.path.join(DATA_DIR, 'train_labels')
        },
        "test": {
            "x": os.path.join(DATA_DIR, 'test'),
            "y": os.path.join(DATA_DIR, 'test_labels')
        },
        "valid": {
            "x": os.path.join(DATA_DIR, 'val'),
            "y": os.path.join(DATA_DIR, 'val_labels')
        }
    }
    return dir[fold]

def load_config():
    working_directory = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(working_directory,".env")):
        config = dotenv_values(os.path.join(working_directory,".env"))
    else:
        config = dotenv_values(os.path.join(working_directory,".env.default"))

    for key, value in config.items():
        if re.match('true', value, re.IGNORECASE):
            config[key] = True
        elif re.match('false', value, re.IGNORECASE):
            config[key] = False
        elif re.match('\d+', value):
            config[key] = int(value)
    return config

import os
import sys
import pathlib

working_directory = pathlib.Path(__file__).parent.parent.resolve()
sys.path.append(str(working_directory))
