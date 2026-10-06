from ..types import Tool
from .bash import bash_tool
from .edit import edit_tool
from .find import find_tool
from .grep import grep_tool
from .ls import ls_tool
from .read import read_tool
from .stat import stat_tool
from .write import write_tool

tools: list[Tool] = [
    bash_tool,
    read_tool,
    write_tool,
    edit_tool,
    ls_tool,
    grep_tool,
    stat_tool,
    find_tool,
]
