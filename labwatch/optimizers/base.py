class Optimizer(object):
    def __init__(self, config_space):
        self.config_space = config_space

    def get_random_config(self):
        return self.config_space.sample()

    def get_default_config(self):
        return self.config_space.default()

    def suggest_configuration(self):
        """
        the default is to return nothing
        this is done such that the user / the watcher
        can then check the queue or try something else
        if you want get_random_config as default
        create an instance of RandomSearch instead!
        """
        return None

    def update(self):
        raise NotImplementedError("update called on base class Optimizer. Use a derived class instead!")

    def needs_updates(self):
        return False

class RandomSearch(Optimizer):

    def __init__(self, config_space):
        super(RandomSearch, self).__init__(config_space)

    def suggest_configuration(self):
        return self.get_random_config()

    def update(self, config, performance):
        pass

    def needs_updates(self):
        return False
