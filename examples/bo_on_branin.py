from branin import ex

print("RUNNING sampled configs")
num_configs = 100
for i in range(num_configs):
    ex.run(named_configs=['search_space'])
