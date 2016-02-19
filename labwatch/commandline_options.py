from sacred.commandline_options import CommandLineOption

class AssistedOption(CommandLineOption):
    """Perform the run in assisted mode, asking the LabAssistant for a configuration."""
    @classmethod
    def apply(cls, args, run):
        # NOTE: this flag is not read by sacred
        #       we just set it so that we can easily identify
        #       assisted runs later, e.g. after running a bunch of runs
        run.assisted = True
