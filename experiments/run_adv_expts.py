#!/usr/bin/env python3
"""Run adversarial robustness experiments (call this once per problem you want
to work on."""

import argparse

# all of the supported domains
DOMAINS = ('adv_cliff', 'adv_gravity', 'adv_race')
STAGES = ('train', 'benchmark', 'mktable', 'mkcurves', 'mkmontage')

parser = argparse.ArgumentParser()
parser.add_argument('stage', choices=STAGES,
        help='stage of training/testing to perform (one of %s)' %
        (', '.join(STAGES), ))
parser.add_argument('domain', choices=DOMAINS,
        help='domain to run on (one of %s)' % (', '.join(DOMAINS), ))
parser.add_argument('--ntrain', default=10,
        help='number of training runs per configuration')


def stage_train(args):
    """Train all configurations."""
    pass


def stage_benchmark(args):
    """Benchmark all configurations on original, noadv, and transfer
    environments."""
    pass


def stage_mktable(args):
    """Make one row out of a table of benchmark results."""
    pass


def stage_mkcurves(args):
    """Make per-method learning curves for report."""
    pass


def stage_mkmontage(args):
    """Make some random rollout montages for each combination of method &
    original/noadv/transfer environment."""
    pass


def main(args):
    stage = args.stage
    stage_func = globals()['stage_' + stage]
    stage_func(args)


if __name__ == '__main__':
    main(parser.parse_args())
