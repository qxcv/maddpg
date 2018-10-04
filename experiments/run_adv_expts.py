#!/usr/bin/env python3
"""Run adversarial robustness experiments (call this once per problem you want
to work on."""

import argparse
import os
import pickle
import subprocess

import tqdm

# all of the supported domains
DOMAINS = ('adv_cliff', 'adv_gravity', 'adv_race')
STAGES = ('train', 'benchmark', 'mktable', 'mkcurves', 'mkmontage')
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(THIS_DIR, 'train.py')

parser = argparse.ArgumentParser()
parser.add_argument(
    'stage',
    choices=STAGES,
    help='stage of training/testing to perform (one of %s)' %
    (', '.join(STAGES), ))
parser.add_argument(
    'domain',
    choices=DOMAINS,
    help='domain to run on (one of %s)' % (', '.join(DOMAINS), ))
parser.add_argument(
    '--ntrain',
    type=int,
    default=10,
    help='number of training runs per configuration')


def launch_train(args):
    cmdline = ['python', TRAIN_SCRIPT] + list(str(a) for a in args)
    print('Spawning', ' '.join(cmdline))
    return subprocess.Popen(
        cmdline,
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)


def wait_all(procs):
    """Wait for all subprocess.Popen objects to terminate. Give warnings etc.
    if things don't go so well."""
    waitlist = list(enumerate(procs))

    with tqdm.trange(len(waitlist)) as pbar:
        while waitlist:
            for i, proc_true_idx in enumerate(waitlist):
                true_idx, proc = waitlist[i]
                try:
                    retcode = proc.wait(0.1)
                except subprocess.TimeoutExpired:
                    continue
                # print warnings for any errors
                if retcode != 0:
                    print('Process %d exited with return code %d' %
                          (true_idx, retcode))
                    print('That was a bad retcode. Check the output.')
                    print('======== stderr ========')
                    print(proc.stderr.read())
                    print('======== stdout ========')
                    print(proc.stdout.read())
                # remove from waitlist & update progress bar
                del waitlist[i]
                pbar.update(1)
                break


def stage_train(args):
    """Train all configurations."""
    # We've already been given a domain. Configurations:
    # - MADDPG
    # - DDDPG with adversary
    # - DDPG without adversary (append _nulladv)
    all_procs = []
    print('Spawning subprocesses')
    for i in range(args.ntrain):
        maddpg_proc = launch_train([
            '--scenario',
            args.domain,
            '--num-adversaries',
            1,
            '--save-root',
            'data/%s/maddpg-%d/' % (args.domain, i),
            '--exp-name',
            'results',
        ])
        ddpg_adv_proc = launch_train([
            '--scenario',
            args.domain,
            '--num-adversaries',
            1,
            '--adv-policy',
            'ddpg',
            '--good-policy',
            'ddpg',
            '--save-root',
            'data/%s/ddpg-only-adv-%d/' % (args.domain, i),
            '--exp-name',
            'results',
        ])
        ddpg_nulladv_proc = launch_train([
            '--scenario',
            args.domain + '_nulladv',
            '--num-adversaries',
            1,
            '--adv-policy',
            'ddpg',
            '--good-policy',
            'ddpg',
            '--save-root',
            'data/%s/ddpg-only-nulladv-%d/' % (args.domain, i),
            '--exp-name',
            'results',
        ])
        all_procs.extend([maddpg_proc, ddpg_adv_proc, ddpg_nulladv_proc])
    print('Waiting for subprocesses to terminate')
    wait_all(all_procs)


def stage_benchmark(args):
    """Benchmark all configurations on original, noadv, and transfer
    environments."""
    raise NotImplementedError()


def stage_mktable(args):
    """Make one row out of a table of benchmark results."""
    raise NotImplementedError()


def stage_mkcurves(args):
    """Make per-method learning curves for report."""
    raise NotImplementedError()


def stage_mkmontage(args):
    """Make some random rollout montages for each combination of method &
    original/noadv/transfer environment."""
    raise NotImplementedError()


def main(args):
    stage = args.stage
    stage_func = globals()['stage_' + stage]
    stage_func(args)


if __name__ == '__main__':
    main(parser.parse_args())
