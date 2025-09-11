
from dataclasses import dataclass
from typing import Optional
import torch
from collections import OrderedDict
import os, glob, random, math, json, typing as T
from pathlib import Path
import numpy as np
import wfdb
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from google.colab import drive
import shutil, os
import os, glob
import os, random, numpy as np, wfdb, torch
from collections import Counter, defaultdict, OrderedDict
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd
from torch.utils.data import ConcatDataset, Subset, random_split, RandomSampler, WeightedRandomSampler
from torch.utils.data.dataset import ConcatDataset, Subset
from torch.utils.data import random_split, RandomSampler, WeightedRandomSampler
import ast
from typing import List, Tuple
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
from math import ceil
from torch.optim.lr_scheduler import LambdaLR
import os, sys
from collections import Counter
from google.colab import drive  # will exist in Colab
from torch.nn.utils import clip_grad_norm_
import math
from pprint import pprint
from collections import defaultdict
from pprint import pprint; pprint(res)
from sklearn.metrics import f1_score
import traceback
import time
from typing import Any, Dict, Tuple, List
from torch.optim import Adam
import traceback; traceback.print_exc()
import pandas as pd, numpy as np, inspect
from caas_jupyter_tools import display_dataframe_to_user
import torch, torch.nn.functional as F
import torch, numpy as np
import csv
import json
from typing import Dict, Tuple, Any, List
import itertools
import math, numpy as np
import pandas as pd, time


import argparse
from tinyml.experiments import *
from tinyml.training import *
from tinyml.models import *
from tinyml.data import *

def _default_entrypoint():
    # Attempt to find a reasonable experiment function to call
    candidates = [name for name, obj in globals().items() if callable(obj) and name.startswith(("run_", "experiment", "main_experiment"))]
    if candidates:
        print(f"Calling: {candidates[0]}()")
        return globals()[candidates[0]]()
    print("No obvious experiment entrypoint found. Please call your experiment function directly.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TinyML Modularized Runner")
    parser.add_argument("--entry", type=str, default="", help="Fully qualified function to call, e.g., tinyml.experiments.run_apnea")
    args = parser.parse_args()

    if args.entry:
        # Dynamically import and call the entry
        mod_name, func_name = args.entry.rsplit(".", 1)
        mod = __import__(mod_name, fromlist=[func_name])
        fn = getattr(mod, func_name)
        fn()
    else:
        _default_entrypoint()
