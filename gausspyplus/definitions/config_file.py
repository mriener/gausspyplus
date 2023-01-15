import ast
import configparser
import os
from dataclasses import fields
from pathlib import Path
from typing import Optional

from astropy import units as u

from gausspyplus.utils.output import say

from gausspyplus.definitions import definitions


def _add_settings_to_config_file(config_header, settings, all_keywords=False, description=True):
    content_for_config_file = [f"\n\n\n[{config_header}]"]
    for setting in settings:
        if setting.metadata["simple"] or all_keywords:
            if description:
                content_for_config_file.append(f"\n\n# {setting.metadata['description']}")
            content_for_config_file.append(f"\n{setting.name} = {setting.default}")
    return content_for_config_file


def make(
    all_keywords: bool = False,
    description: bool = True,
    output_directory: Optional[Path] = None,
    filename: str = "gausspy+.ini",
) -> None:
    """Create a GaussPy+ configuration file.

    Parameters
    ----------
    all_keywords : bool
        Default is `False`, which includes only the most essential parameters. If set to `True`, include all parameters in the configuration file.
    description : bool
        Default is `True`, which includes descriptions of the parameters in the configuration file.
    output_directory : string
        Directory to which configuration file gets saved.
    filename : string
        Name of the configuration file.

    """
    config_file = ["# Configuration file for GaussPy+"]

    for config_header, setting in {
        "DEFAULT": definitions.SettingsDefault,
        "training": definitions.SettingsTraining,
        "preparation": definitions.SettingsPreparation,
        "decomposition": definitions.SettingsDecomposition,
        "spatial fitting": definitions.SettingsSpatialFitting,
    }.items():

        config_file += _add_settings_to_config_file(
            config_header=config_header,
            settings=fields(setting),
            all_keywords=all_keywords,
            description=description,
        )

    if not output_directory:
        output_directory = Path(os.getcwd())

    with open(output_directory / filename, "w") as file:
        for line in config_file:
            file.write(line)
        say(f"'{filename}' in '{output_directory}'", task="save")


def get_values_from_config_file(self, config_file, config_key="DEFAULT"):
    """Read in values from a GaussPy+ configuration file.

    Parameters
    ----------
    config_file : str
        Filepath to configuration file of GaussPy+.
    config_key : str
        Section of GaussPy+ configuration file, whose parameters should be read in addition to 'DEFAULT'.

    """
    config = configparser.ConfigParser()
    config.read(config_file)

    for key, value in config[config_key].items():
        try:
            setattr(self, key, ast.literal_eval(value))
        except ValueError as e:
            if key != "vel_unit":
                raise Exception(f"Could not parse parameter {key} from config file") from e
            value = u.Unit(value)
            setattr(self, key, value)
