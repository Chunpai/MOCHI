import os
import logging
import pathlib


def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return:
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
    except Exception as err:
        logging.getLogger("Dirs Creator").info("Creating directories error: {0}".format(err))
        exit(-1)


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
