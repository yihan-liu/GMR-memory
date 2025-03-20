# TODO: Plot all samples in a scatter plot.
# X axis shows the time passed since last event (placement/removal),
# Y axis shows the time predicted by the model
# Use different colors to show placement and removal.
# TODO: (PROBABLY) use different colors for different rounds of tests

from gmr.gmr_memory_dataset import GMRMemoryDataset
from gmr.model import GMRMemoryModelDualHead
from gmr.utils import *