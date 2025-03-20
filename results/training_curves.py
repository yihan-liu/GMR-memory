import os
import numpy as np
import matplotlib.pyplot as plt

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

phase1_report = np.load(os.path.join(__location__, 'phase1_report.npy'))
phase2_report = np.load(os.path.join(__location__, 'phase2_report.npy'))

plt.plot(phase1_report)
plt.show() 