from json import load
import os
from dotenv import dotenv_values
import numpy as np
import re
import pathlib

def load_config():
    working_directory = pathlib.Path(__file__).parent.parent.resolve()
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

