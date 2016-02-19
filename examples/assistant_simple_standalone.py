# import the assistant
from simple import a

# let's run 5 random configs
print("RUNNING sampled configs")
num_configs = 5
for i in range(num_configs):
    a.run_suggestion()

print("RUNNING random configs")
for i in range(num_configs):
    a.run_random()

print("RUNNING default config")
a.run_default()
