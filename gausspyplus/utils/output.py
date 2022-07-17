import os
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, Union


def check_if_value_is_none(
        condition,
        value,
        varname_condition,
        varname_value,
        additional_text=''
):
    """Raise error message if no value is supplied for a selected condition.

    The error message is raised if the condition is 'True' and the value is 'None'.

    Parameters
    ----------
    condition : bool
        Selected condition.
    value : type
        Value for the condition.
    varname_condition : str
        Variable name of `condition`.
    varname_value : str
        Variable name of `value`.

    """
    if condition and (value is None):
        raise Exception(f"Need to specify '{varname_value}' for '{varname_condition}'=True. {additional_text}")


def check_if_all_values_are_none(value1, value2, varname_value1, varname_value2):
    # TODO: refactor check_if_all_values_are_none with f strings to avoid repeated variable name
    """Raise error message if both values are 'None'.

    Parameters
    ----------
    value1 : type
        Description of parameter `value1`.
    value2 : type
        Description of parameter `value2`.
    varname_value1 : str
        Variable name of `value1`.
    varname_value2 : str
        Variable name of `value2`.

    """
    if (value1 is None) and (value2 is None):
        raise Exception(f"Need to specify either '{varname_value1}' or '{varname_value2}'.")


def set_up_logger(
        parentDirname: Union[str, Path],
        filename: str,
        method: str = 'g+_decomposition'
) -> logging.Logger:
    #  setting up logger
    now = datetime.now()
    date_string = "{}{}{}-{}{}{}".format(
        now.year,
        str(now.month).zfill(2),
        str(now.day).zfill(2),
        str(now.hour).zfill(2),
        str(now.minute).zfill(2),
        str(now.second).zfill(2))

    dirname = os.path.join(parentDirname, 'gpy_log')
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    filename = os.path.splitext(os.path.basename(filename))[0]

    logname = os.path.join(dirname, f'{date_string}_{method}_{filename}.log')
    logging.basicConfig(filename=logname,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    return logging.getLogger(__name__)


def make_pretty_header(string: str) -> str:
    """Return a nicely formatted heading."""
    line = '=' * len(string)
    return f"\n{line}\n{string}\n{line}"


def say(message: str,
        task: Optional[Literal["save"]] = None,
        verbose: bool = True,
        logger: bool = False,
        end: Optional[str] = None) -> None:
    """Diagnostic messages."""
    note_prefix = {"save": "SAVED FILE: ",
                   None: ""}
    stdout_format = {"save": lambda msg: f"\033[92m{note_prefix['save']}\033[0m{msg}",
                     None: lambda msg: msg}

    if logger:
        logger.info(f"{note_prefix[task]} {message}")
    if verbose:
        print(stdout_format[task](message), end=end)


def format_warning(message, category, filename, lineno, file=None, line=None):
    sys.stderr.write("\n\033[93mWARNING:\033[0m {}: {}\n".format(
        category.__name__, message))
