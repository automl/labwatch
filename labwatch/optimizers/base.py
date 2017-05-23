
class Optimizer(object):
    """Defines the interface for all optimizers."""

    def __init__(self, config_space):
        self.config_space = config_space

    def get_random_config(self):
        return self.config_space.sample()

    def get_default_config(self):
        return self.config_space.default()

    def suggest_configuration(self):
        """Suggests a configuration of hyperparameters to be run.

        Returns
        -------
        dict:
            Dictionary mapping parameter names to suggested values.

            The default is to return nothing this is done such that the
            user / the watcher can then check the queue or try something else.
            If you want get_random_config as default create an instance of
            RandomSearch instead.
        """
        return None

    def update(self, configs, costs, runs):
        """
        Update the internal state of the optimizer with a list of new results.

        Parameters
        ----------
        configs: list[dict]
            List of configurations mapping parameter names to values.
        costs: list[float]
            List of costs associated to each config.
        runs: list[dict]
            List of dictionaries containing additional run information.
        """
        raise NotImplementedError("update called on base class Optimizer. "
                                  "Use a derived class instead!")

    def needs_updates(self):
        """
        Returns
        -------
        bool:
            True if this optimizer needs updates, False otherwise.
        """
        return False
