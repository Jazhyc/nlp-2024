"""
This file contains some utility functions like parsing the config file, etc.
"""

import re

def parse_sh_args(filepath):
    """Parses a shell script line into a dictionary of arguments.

    Args:
        sh_line: The shell script line to parse.

    Returns:
        A dictionary containing the parsed arguments.
    """

    # Read whole file
    with open(filepath, "r") as file:
        sh_line = file.read()
        
    # Skip comments
    sh_line = re.sub(r"^\s*#.*$", "", sh_line, flags=re.MULTILINE)

    # Parse arguments, format is --name value
    # if no value is provided (\ is observed), it is set to True
    args = {}
    for match in re.finditer(r"--(\S+)(?:\s+(\S+))?", sh_line):
        name, value = match.groups()
        args[name] = value if value is not None else True

        # replace \\ with True boolean
        if args[name] == "\\":
            args[name] = True

        # Convert to number if possible
        try:
            args[name] = int(args[name])
        except ValueError:
            try:
                args[name] = float(args[name])
            except ValueError:
                pass

    return args

if __name__ == "__main__":
  print(parse_sh_args("../fine_tune_rag.sh"))