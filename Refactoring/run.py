import source.developed as developed
developed.print_stamp()

import argparse

import matplotlib.pyplot as plt
import tensorflow as tf

# custom modules


def main():

    print("main")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--make', type=bool, default=False, help='Default: False. Enter True to update the dataset.')
    parser.add_argument('--boost', type=int, default=0, help='Default: 0. ')
    parser.add_argument('--batch', type=int, default=10, help='Default: 10. Batches per iteration, the number of data to be training and testing.')
    parser.add_argument('--epochs', type=int, default=100, help='Default: 100')
    parser.add_argument('--validation', type=int, default=0, help='Default: 0')
    FLAGS, unparsed = parser.parse_known_args()

    main()
