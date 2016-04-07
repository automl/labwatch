'''
Created on Feb 24, 2016

@author: Aaron Klein
'''

from labwatch.assistant import LabAssistant
from labwatch.optimizers import SMAC3

from branin import db
from branin import ex


a = LabAssistant(db, ex, optimizer=SMAC3, always_inject_observer=True)

print("RUNNING sampled configs")
num_configs = 100
for i in range(num_configs):
    a.run_suggestion(command="branin_cost")
