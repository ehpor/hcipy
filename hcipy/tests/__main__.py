import sys

from .runner import run_tests_cli

if __name__ == '__main__':
    sys.exit(run_tests_cli(sys.argv[1:]))
