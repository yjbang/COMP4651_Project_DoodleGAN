import argparse


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    argparser.add_argument(
        '--resume',
        default=None,
        action='store_true',
        help='resume from checkpoint')
    argparser.add_argument(
        '--mode',
        default='train',
        choices=['train', 'test'],
        help='choose between training and testing mode')
    argparser.add_argument(
        '--data',
        default='drawit',
        choices=['drawit', 'mnist'],
        help='choose between training and testing mode')
    args = argparser.parse_args()
    return args
