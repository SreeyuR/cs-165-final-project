import os
import sys
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig


def load_config(config_folder, config_file, config_name, verbose=False):
    """
    Loads configuration from a YAML file and command-line arguments.

    This function reads configuration settings from a YAML file located in the
    specified folder and merges them with any provided command-line arguments.
    If the `verbose` option is set, it prints the configuration details.

    Parameters:
    ----------
    config_folder : str
       Directory containing the configuration files.
    config_file : str
       The primary YAML configuration file.
    config_name : str
       Identifier for the specific configuration settings within the YAML file.

    Returns:
    -------
    config : object
       The merged configuration object.
    """
    pipe = ConfigPipeline(
        [
            YamlConfig(
                config_file, config_name=config_name, config_folder=config_folder
            ),
            ArgparseConfig(infer_types=True, config_name=None, config_file=None),
            YamlConfig(config_folder=config_folder),
        ]
    )
    config = pipe.read_conf()
    # Print config to screen
    if verbose:
      pipe.log()
      sys.stdout.flush()
    return config

