import sys
import argparse

print(sys.argv)

parser = argparse.ArgumentParser()
parser.add_argument("--num", type=int)

cli_args = parser.parse_args(sys.argv[1:])
print(cli_args)
