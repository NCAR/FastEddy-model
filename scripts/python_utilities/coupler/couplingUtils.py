import argparse

def parse_args():
    """ parse the command line arguments """

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True, help="JSON file with coupler parameter settings")
    args = parser.parse_args()
    return args
