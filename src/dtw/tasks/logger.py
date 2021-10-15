#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#       File author(s):
#           Jonathan Legrand <jonathan.legrand@ens-lyon.fr>
#
#       File maintainer(s):
#           Jonathan Legrand <jonathan.legrand@ens-lyon.fr>
#
#       Mosaic Inria team, RDP Lab, Lyon
# ------------------------------------------------------------------------------

import logging
import sys

DEFAULT_LOG_LEVEL = 'INFO'
LOG_FMT = ("%(name)s - l.%(lineno)d - %(levelname)s: %(message)s",)
BIN_LOG_FMT = "%(asctime)s - %(levelname)s: %(message)s", '%Y-%m-%d %H:%M:%S'

def get_console_handler(fmt):
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(*fmt)
    console_handler.setFormatter(formatter)
    return console_handler

def get_file_handler(log_file, log_level, fmt):
    file_handler = logging.FileHandler(log_file, mode='w')
    formatter = logging.Formatter(*fmt)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)
    return file_handler

def get_logger(logger_name, log_file=None, log_level=None, fmt=LOG_FMT):
    logger = logging.getLogger(logger_name)

    if log_level is not None:
        logger.setLevel(log_level)

    logger.addHandler(get_console_handler(fmt))
    if log_level is not None and log_file is not None:
        logger.addHandler(get_file_handler(log_file, log_level, fmt))
    logger.propagate = False

    return logger
