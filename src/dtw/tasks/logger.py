#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import sys

DEFAULT_LOG_LEVEL = 'INFO'
LOG_FMT = "%(name)s - %(lineno)d - %(levelname)s - %(message)s"
FORMATTER = logging.Formatter(LOG_FMT)

def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler

def get_file_handler(log_file, log_level):
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(FORMATTER)
    file_handler.setLevel(log_level)
    return file_handler

def get_logger(logger_name, log_file, log_level):
    logger = logging.getLogger(logger_name)

    logger.setLevel(log_level)

    logger.addHandler(get_console_handler())
    logger.addHandler(get_file_handler(log_file, log_level))
    logger.propagate = False

    return logger
