import click

from rich import print

from jarvais import __version__

from . import set_log_verbosity
from .analyzer import analyzer


@click.group(no_args_is_help=True)
@set_log_verbosity()
@click.version_option(
    version=__version__,
    package_name="jarvais",
    prog_name="jarvais",
    message="%(package)s:%(prog)s:%(version)s",
)
@click.help_option("-h", "--help")
def cli(verbose: int, quiet: bool) -> None:
    """JARVAIS CLI - ML toolkit for oncology workflows."""
    pass

cli.add_command(analyzer)

if __name__ == "__main__":
    cli()