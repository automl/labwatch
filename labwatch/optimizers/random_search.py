
from labwatch.optimizers import Optimizer


class RandomSearch(Optimizer):

    def __init__(self, config_space):
        super(RandomSearch, self).__init__(config_space)

    def suggest_configuration(self):
        return self.get_random_config()

    def update(self, configs, costs, run_info):
        pass

    def needs_updates(self):
        return False
