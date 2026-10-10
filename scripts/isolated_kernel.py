"""Research kernel entry point: never load host IPython configuration or startup code."""
from pathlib import Path
import sys

from ipykernel.kernelapp import IPKernelApp


class ResearchKernelApp(IPKernelApp):
    def load_config_file(self, *args, **kwargs):
        # Colab's system config includes both a custom kernel and Google extensions.
        # A venv does not isolate these config files. Do not execute them at all.
        pass

    def init_code(self):
        # Research code enters only through the reviewed notebook cells.
        pass


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    ResearchKernelApp.launch_instance()
