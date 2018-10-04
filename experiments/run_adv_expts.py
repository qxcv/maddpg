#!/usr/bin/env python3
"""Run adversarial robustness experiments (call this once per problem you want
to work on."""

import argparse
import os
import pickle
import subprocess

import numpy as np
import skimage.io
import skimage.morphology
import skvideo.io
import tqdm

# all of the supported domains
DOMAINS = ('adv_cliff', 'adv_gravity', 'adv_race')
STAGES = ('train', 'benchmark', 'movie',
          'mktables', 'mkcurves', 'mkcomposites')
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


def launch_train(args, *, xvfb=False):
    cmdline = ['python', TRAIN_SCRIPT] + list(str(a) for a in args)
    if xvfb:
        # use virtual X server
        cmdline = ['xvfb-run', '-a', '-s', '-screen 0 1400x1400x24'] + cmdline
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
    all_procs = []
    for i in range(args.ntrain):
        for expt in 'nulladv', 'transfer':
            suff = 'maddpg_policy/snapshot-49200'
            dom = args.domain
            maddpg_proc = launch_train([
                '--scenario',
                dom + '_' + expt,
                '--num-adversaries',
                1,
                '--save-root',
                'data/%s/maddpg-%d/' % (dom, i),
                '--exp-name',
                'results_' + expt,
                '--load',
                'data/%s/maddpg-%d/%s' % (dom, i, suff),
                '--benchmark'
            ])
            ddpg_adv_proc = launch_train([
                '--scenario',
                dom + '_' + expt,
                '--num-adversaries',
                1,
                '--adv-policy',
                'ddpg',
                '--good-policy',
                'ddpg',
                '--save-root',
                'data/%s/ddpg-only-adv-%d/' % (dom, i),
                '--exp-name',
                'results_' + expt,
                '--load',
                'data/%s/ddpg-only-adv-%d/%s' % (dom, i, suff),
                '--benchmark'
            ])
            ddpg_nulladv_proc = launch_train([
                '--scenario',
                dom + '_' + expt,
                '--num-adversaries',
                1,
                '--adv-policy',
                'ddpg',
                '--good-policy',
                'ddpg',
                '--save-root',
                'data/%s/ddpg-only-nulladv-%d/' % (dom, i),
                '--exp-name',
                'results_' + expt,
                '--load',
                'data/%s/ddpg-only-nulladv-%d/%s' % (dom, i, suff),
                '--benchmark'
            ])
            all_procs.extend([maddpg_proc, ddpg_adv_proc, ddpg_nulladv_proc])
    wait_all(all_procs)


def stage_movie(args):
    """Make movies for a few episodes in each test (and train
    configuration)."""
    all_procs = []
    for i in range(args.ntrain):
        for expt in '', '_nulladv', '_transfer':
            suff = 'maddpg_policy/snapshot-49200'
            dom = args.domain
            maddpg_proc = launch_train([
                '--scenario',
                dom + expt,
                '--num-adversaries',
                1,
                '--save-root',
                'data/%s/maddpg-%d/' % (dom, i),
                '--exp-name',
                'results' + expt,
                '--load',
                'data/%s/maddpg-%d/%s' % (dom, i, suff),
                '--movie'
            ], xvfb=True)
            ddpg_adv_proc = launch_train([
                '--scenario',
                dom + expt,
                '--num-adversaries',
                1,
                '--adv-policy',
                'ddpg',
                '--good-policy',
                'ddpg',
                '--save-root',
                'data/%s/ddpg-only-adv-%d/' % (dom, i),
                '--exp-name',
                'results' + expt,
                '--load',
                'data/%s/ddpg-only-adv-%d/%s' % (dom, i, suff),
                '--movie'
            ], xvfb=True)
            ddpg_nulladv_proc = launch_train([
                '--scenario',
                dom + expt,
                '--num-adversaries',
                1,
                '--adv-policy',
                'ddpg',
                '--good-policy',
                'ddpg',
                '--save-root',
                'data/%s/ddpg-only-nulladv-%d/' % (dom, i),
                '--exp-name',
                'results' + expt,
                '--load',
                'data/%s/ddpg-only-nulladv-%d/%s' % (dom, i, suff),
                '--movie'
            ], xvfb=True)
            all_procs.extend([maddpg_proc, ddpg_adv_proc, ddpg_nulladv_proc])
            # HACK rate-limiting so I can run this on my crummy laptop
            print('\n\nWAITING\n\n')
            all_procs[-1].wait()
    wait_all(all_procs)


def stage_mktables(args):
    """Make one row out of a table of benchmark results."""
    raise NotImplementedError()


def stage_mkcurves(args):
    """Make per-method learning curves for report."""
    raise NotImplementedError()


def blend_frames(frames, subsample=1):
    """Blend together a T*H*W*C (type uint8) volume of video frames taken in
    front of a static background. Should yield decent results for my
    environments, but no guarantees outside of there."""
    if subsample:
        # subsample in convoluted way to ensure we always get last frame
        frames = frames[::-1][::subsample][::-1]
    med_frame = np.median(frames, axis=0)
    # our job is to find weights for frames st frames average out in the end
    frame_weights = np.zeros(frames.shape[:3] + (1,))
    for frame_idx, frame in enumerate(frames):
        pixel_dists = np.linalg.norm(frame - med_frame, axis=2)
        diff_pixel_mask = pixel_dists > 0.05
        # fade in by a few pixels
        for i in range(4):
            eroded = skimage.morphology.erosion(diff_pixel_mask)
            diff_pixel_mask = 0.5 * eroded + 0.5 * diff_pixel_mask
        frame_weights[frame_idx, :, :, 0] = diff_pixel_mask
    # give later frames a bonus for blending over top of others
    # (edit: removed this because it led to ugly artefacts in some places)
    # frame_range = np.arange(len(frames)) \
    #     .reshape(-1, 1, 1, 1).astype('float32')
    # frame_weights *= 1 + frame_range
    # normalise frame weights while avoiding division by zero
    frame_weight_sums = frame_weights.sum(axis=0)[None, ...]
    frame_weights = np.divide(
        frame_weights,
        frame_weight_sums,
        where=frame_weight_sums > 0)
    # now denormalize so that later frames get brighter than earlier
    # ones
    n = len(frame_weights)
    min_alpha = 0.6
    age_descale = min_alpha + (1 - min_alpha) * np.arange(n) / (n - 1)
    age_descale = age_descale.reshape((-1, 1, 1, 1))
    frame_weights = frame_weights * age_descale
    frame_weight_sums = frame_weights.sum(axis=0)
    # finally blend
    combined_frames = np.sum(frames * frame_weights, axis=0)
    combined_frames += med_frame * (1 - frame_weight_sums)
    # clip bad pixel values
    combined_frames[combined_frames < 0] = 0
    combined_frames[combined_frames > 1] = 1
    return combined_frames


def load_video(vid_path):
    byte_vid = skvideo.io.vread(vid_path)
    float_vid = byte_vid.astype('float32') / 255.0
    return float_vid


def stage_mkcomposites(args):
    """Make some random rollout composites for each combination of method &
    original/noadv/transfer environment."""
    # first find all relevant videos with a directory walk
    vid_paths = []
    for root_dir, _, filenames in os.walk('./data/'):
        basename = os.path.basename(root_dir.rstrip(os.sep))
        if basename != 'movie_files':
            continue
        for filename in filenames:
            if not filename.endswith('.mp4'):
                continue
            vid_path = os.path.join(root_dir, filename)
            vid_paths.append(vid_path)
    # now blend each of the videos and place a new image in corresponding
    # directory
    print('Found %d videos; will write composites now' % len(vid_paths))
    for vid_num, vid_path in enumerate(vid_paths, start=1):
        video = load_video(vid_path)
        composite = blend_frames(video, subsample=3)
        prefix, _ = os.path.splitext(vid_path)
        out_path = prefix + '_composite.png'
        print('[%d/%d] Writing %s' % (vid_num, len(vid_paths), out_path))
        skimage.io.imsave(out_path, composite)


def main(args):
    stage = args.stage
    stage_func = globals()['stage_' + stage]
    stage_func(args)


if __name__ == '__main__':
    main(parser.parse_args())
