import os
import shutil
import pickle
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN
import numpy as np
import random
import torch


def round_decimal_percentage(value, decimal=2):
    """Round to 2 decimal of percentage
       0.9652132 to 96.52 % """
    decimal_zeros = ''
    for _ in range(decimal):
        decimal_zeros = '0' + decimal_zeros
    new_value_str = str(
        Decimal(str(value * 100)).quantize(Decimal('0.{}'.format(decimal_zeros)), rounding=ROUND_HALF_EVEN)) + "%"
    return new_value_str


def round_decimal(value, decimal=4):
    """Round to 2 decimal
       0.9652132 to 0.9652 """
    decimal_zeros = ''
    for _ in range(decimal):
        decimal_zeros = '0' + decimal_zeros

    new_value_str = str(
        Decimal(str(value)).quantize(Decimal('0.{}'.format(decimal_zeros)), rounding=ROUND_HALF_EVEN))

    return new_value_str


def round_decimal_iterable(iter, decimal=3):
    rounded_iter = list(map(round_decimal, iter, np.repeat(decimal, len(iter))))
    return rounded_iter


def makepath(dir):
    try:
        os.makedirs(dir)
    except OSError:
        pass


def unify_path(path):
    """Remove '/' at the end, if it exists"""
    if path[-1] == '/':
        path = path[:-1]
    else:
        path = path

    return path


def fix_seed(manualSeed):
    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)   # seed for cpu
    torch.cuda.manual_seed(manualSeed)  # seed for gpu
    torch.cuda.manual_seed_all(manualSeed)  # seed for all gpu



