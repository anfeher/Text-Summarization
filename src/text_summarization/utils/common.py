import os
import yaml

from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations
from pathlib import Path

from text_summarization.logging import logger

@ensure_annotations
def read_yaml(path_to_yaml:Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (Path): path like input

    Raises:
        ValueError: if yaml file is empty
        e: handle other errors

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    
@ensure_annotations
def create_directories(path_to_directories:list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        verbose (bool, optional): if True print creation logs. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directorie ar: {path}")

@ensure_annotations
def get_size(path:Path) -> str:
    """get size in kB

    Args:
        path (Path): path to the file

    Returns:
        str: size in kB
    """
    size_in_kB = round(os.path.getsize(path)/1024)
    return f"~{size_in_kB} kB"