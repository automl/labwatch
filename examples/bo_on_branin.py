from branin import a

print("RUNNING sampled configs")
num_configs = 100
for i in range(num_configs):
    a.run_suggestion(command="branin")
