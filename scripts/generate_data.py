import yaml
import numpy as np
import os
from sbi import Simulator, PROJECT_ROOT
import argparse
parser = argparse.ArgumentParser(description="Generate data based on a configuration file.")
parser.add_argument("--config", type=str, required=True, help="Path to the configuration YAML file.")
args = parser.parse_args()

with open(args.config, "r") as file:
    config = yaml.safe_load(file)

parameters = np.array([[config["parameters"]["omega"], 
                        config["parameters"]["phi"], 
                        config["parameters"]["A"]]])

t = np.linspace(config["time_steps"]["min"], config["time_steps"]["max"], config["time_steps"]["n"])
simulator = Simulator(config["sigma"], t ) 
data = simulator.simulate(parameters)
np.save(os.path.join(PROJECT_ROOT,"data", "observed_data.npy"), data)
