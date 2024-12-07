# ======================================================
#       Implementation of Text Generator using LSTM     
# ======================================================

import torch
import torch.nn as nn
import string
import random
import sys
import unidecode
from torch.utils.tensorboard import SummaryWriterr

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Get charecters from string.printable
all_charecters = string.printable
n_charecters = len(all_charecters)

# Read large text file