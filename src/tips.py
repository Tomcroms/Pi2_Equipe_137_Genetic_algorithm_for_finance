import numpy as np
import pandas as pd
import json


d = json.load(open("chemin"))   #convertit json en dictionnaire python

d = {"actionA": 22.34}


class Stock:
    def __init__(self, name, expected_return, std_dev):
        self.name = name
        self.expected_return = expected_return  # Rendement attendu
        self.std_dev = std_dev  # Écart-type du rendement

stockConfig = {"name": "ActionB", "expected_return": 0.6, "std_dev": 22}
s = Stock(**stockConfig)    #instanciation immédiate pour actionB
