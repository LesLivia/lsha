import math
import os
from enum import Enum
from typing import List

import biosignalsnotebooks as bsnb
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy import signal
from scipy.signal import periodogram

import mgrs.emg_mgr as emg_mgr


class Emg:
    def __init__(self, t: float, val: float):
        self.t = t
        self.val = val

    def __str__(self):
        formatted_t = '{:.15f}'.format(self.t)
        if int(self.t) < 10:
            formatted_t = '00' + formatted_t
        elif 10 <= int(self.t) < 100:
            formatted_t = '0' + formatted_t

        return '{}\t{:+.17f}\n'.format(formatted_t, self.val)


class Group(Enum):
    YOUNG = 1
    ELDER = 2

    @staticmethod
    def int_to_group(x: int):
        return Group.YOUNG if x == 1 else Group.ELDER

    def to_char(self):
        return 'y' if self == Group.YOUNG else 'e'

    def to_str(self):
        return 'Young' if self == Group.YOUNG else 'Elderly'


class Muscles(Enum):
    LEFT_VASTUS_LATERALIS = 0
    BICEPS_FEMORIS = 1
    GASTROCNEMIUS = 2
    TIBIALIS_ANTERIOR = 3


class Mode(Enum):
    WALKING = 1
    RESTING = 0
    ERROR = -1

    @staticmethod
    def int_to_mode(x: int):
        if x == 1:
            return Mode.WALKING
        elif x == 0:
            return Mode.RESTING
        else:
            return Mode.ERROR

    def to_char(self):
        if self == Mode.WALKING:
            return 'w'
        elif self == Mode.RESTING:
            return 'r'
        else:
            return 'e'


class Trial:
    def __init__(self, group: Group, sub_id: int, trial_id: int, vel: int, emg: List[Emg] = None, mode: Mode = None):
        self.group = group
        self.sub_id = sub_id
        self.trial_id = trial_id
        self.vel = vel
        self.emg = emg if emg is not None else []
        self.mode = mode

    def set_emg(self, emg: List[Emg]):
        self.emg = emg

    def set_mode(self, mode: Mode):
        self.mode = mode

    @staticmethod
    def parse_line(line: str):
        fields = line.split('	')
        group = Group.int_to_group(int(fields[1]))
        sub_id = int(fields[0][1:])
        trial_id = int(fields[3])
        vel = int(fields[2])
        mode = Mode.int_to_mode(int(fields[4])) if len(fields) > 4 else None
        return Trial(group, sub_id, trial_id, vel, [], mode)

    def __str__(self):
        group_char = self.group.to_char()
        return 'Subject {}{}, trial {} (vel {})'.format(group_char, self.sub_id, self.trial_id, self.vel)


MUSCLE = Muscles.TIBIALIS_ANTERIOR
SAMPLING_RATE = 1080
LIM = int(5 * 60 * SAMPLING_RATE)


def acquire_trials_list(path: str):
    with open(path) as source:
        lines = source.readlines()
        trials = [Trial.parse_line(line) for line in lines]

    return trials


def load_emg_signal(path: str, t: Trial):
    trial = t.trial_id
    group_char = t.group.to_char()
    df = scipy.io.loadmat('{}/{}{}/rawdata.mat'.format(path, group_char, t.sub_id))
    mask = scipy.io.loadmat('{}/{}{}/spikeindicator.mat'.format(path, group_char, t.sub_id))

    try:
        to_use = mask['trial{}sd'.format(trial)][:, MUSCLE.value].astype(np.bool)
    except IndexError:
        to_use = mask['trial{}sd'.format(trial)][:, 0].astype(np.bool)

    try:
        signal_mv = df['trial{}'.format(trial)][~to_use, MUSCLE.value]
        signal_mv = signal_mv[:LIM]
    except KeyError:
        signal_mv = []

    time = bsnb.generate_time(signal_mv, SAMPLING_RATE)
    emg_data = [Emg(time[index], x) for (index, x) in enumerate(signal_mv)]
    t.set_emg(emg_data)
    return t


def fill_emg_signals(path: str, trials: List[Trial], dump=True):
    for t in trials:
        trial = t.trial_id
        group_char = t.group.to_char()
        print('Processing Subject {}{} trial {}...'.format(group_char, t.sub_id, trial))

        t = load_emg_signal(path, t)

        if os.path.isfile('{}/dump/{}{}/trial{}.txt'.format(path, group_char, t.sub_id, trial)) and os.path.getsize(
                '{}/dump/{}{}/trial{}.txt'.format(path, group_char, t.sub_id, trial)) > 0:
            print('Already dumped')
            continue
        if dump:
            print('Dumping...')
            if not os.path.isdir('{}/dump/{}{}'.format(path, group_char, t.sub_id)):
                os.makedirs('{}/dump/{}{}'.format(path, group_char, t.sub_id))
            txt_file = open('{}/dump/{}{}/trial{}.txt'.format(path, group_char, t.sub_id, trial), 'w')
            txt_file.writelines([str(emg) for emg in t.emg])
            txt_file.close()

        print('Done')

    return trials


def process_trial(trial: Trial, dump=False, cf=0):
    try:
        signal = [x.val for x in trial.emg]
        mean_freq_data = emg_mgr.calculate_mnf(signal, SAMPLING_RATE, cf)

        b_s, b_e = emg_mgr.get_bursts(signal, SAMPLING_RATE)
        bursts = b_e / SAMPLING_RATE
        q, m, x, est_values = emg_mgr.mnf_lin_reg(mean_freq_data, bursts, plot=False)

        if dump:
            new_file = open('resources/hrv_pg/dryad_data/walking_speeds_new.txt', 'a')
            group_char = trial.group.to_char()
            mode = Mode.RESTING if m >= 0 else Mode.WALKING
            padded_id = str(trial.sub_id) if trial.sub_id >= 10 else '0' + str(trial.sub_id)
            to_write = '{}{}\t{}\t{}\t{}\t{}\n'.format(group_char, padded_id, trial.group.value, trial.vel,
                                                       trial.trial_id,
                                                       mode.value)
            new_file.write(to_write)
            new_file.close()

        print('ESTIMATED RATE: {:.6f}'.format(float(m)))
        return m
    except IOError:
        print('An error occurred')


def prog_trial_proc(trial: Trial, initial_guess=None, cf=0):
    signal_mv = [x.val for x in trial.emg]

    b_s, b_e = emg_mgr.get_bursts(signal_mv, SAMPLING_RATE)
    est_lambdas = []
    for i in range(len(b_e)):
        try:
            subset_b_s, subset_b_e = b_s[:i], b_e[:i]

            mean_freq_data = []
            for (index, start) in enumerate(subset_b_s):
                emg_pts = signal_mv[start: subset_b_e[index]]
                freqs, power = periodogram(emg_pts, fs=SAMPLING_RATE)
                # MNF
                try:
                    mnf = sum(freqs * power) / sum(power)
                    mean_freq_data.append(math.log(mnf))
                except ZeroDivisionError:
                    print('error in division')

            # First, design the Buterworth filter
            N = 3  # Filter order
            Wn = 0.3  # Cutoff frequency
            B, A = signal.butter(N, Wn, output='ba')
            smooth_data = signal.filtfilt(B, A, mean_freq_data)
            mean_freq_data = [i * (1 - cf * index) for (index, i) in enumerate(smooth_data)]

            bursts = subset_b_e / SAMPLING_RATE
            q, m, x, est_values = emg_mgr.mnf_lin_reg(mean_freq_data, bursts, plot=False)
            if m < 0:
                est_lambda = math.fabs(m)
                MET = math.log(1 - 0.05) / -est_lambda
                print('ESTIMATED RATE: {:.6f}, MET: {:.2f}min'.format(est_lambda, MET))
            else:
                est_lambda = initial_guess
            est_lambdas.append(est_lambda)
        except ValueError:
            print('insufficient bursts ({} of {})'.format(i, len(b_e)))

    plt.figure(figsize=(10, 5))
    plt.plot(est_lambdas)

    avg_lambdas = []
    for i in range(len(est_lambdas)):
        try:
            avg = sum(est_lambdas[:i]) / i
            avg_lambdas.append(avg)
        except ZeroDivisionError:
            avg_lambdas.append(initial_guess)
    plt.plot(avg_lambdas, 'r')
    plt.show()

    F = [1 - math.exp(-l * t) for (t, l) in enumerate(est_lambdas)]
    F_avg = [1 - math.exp(-l * t) for (t, l) in enumerate(avg_lambdas)]
    F_init = [1 - math.exp(-initial_guess * t) for (t, l) in enumerate(avg_lambdas)]
    F_post = [1 - math.exp(-avg_lambdas[-1] * t) for (t, l) in enumerate(avg_lambdas)]
    print(F_post)
    plt.figure(figsize=(10, 5))
    plt.plot(F, label='instantaneous l')
    plt.plot(F_avg, 'r', label='avg. l')
    plt.plot(F_init, 'g', label='initial guess')
    plt.plot(F_post, 'k', label='post. l')
    plt.plot()
    plt.legend()
    plt.show()


def prog_trial_proc_tpoll(trial: Trial, t_poll: float, initial_guess=None, cf=0):
    signal_mv = [x.val for x in trial.emg]
    est_lambdas = []

    for i in np.arange(0, len(signal_mv), t_poll * SAMPLING_RATE):
        b_s, b_e = emg_mgr.get_bursts(signal_mv[:i], SAMPLING_RATE)
        try:
            mean_freq_data = []
            for (index, start) in enumerate(b_s):
                emg_pts = signal_mv[start: b_e[index]]
                freqs, power = periodogram(emg_pts, fs=SAMPLING_RATE)
                # MNF
                try:
                    mnf = sum(freqs * power) / sum(power)
                    mean_freq_data.append(math.log(mnf))
                except ZeroDivisionError:
                    print('error in division')

            # First, design the Buterworth filter
            N = 3  # Filter order
            Wn = 0.3  # Cutoff frequency
            B, A = signal.butter(N, Wn, output='ba')
            smooth_data = signal.filtfilt(B, A, mean_freq_data)
            mean_freq_data = [i * (1 - cf * index) for (index, i) in enumerate(smooth_data)]

            bursts = b_e / SAMPLING_RATE
            q, m, x, est_values = emg_mgr.mnf_lin_reg(mean_freq_data, bursts, plot=False)
            if m < 0:
                est_lambda = math.fabs(m)
                MET = math.log(1 - 0.05) / -est_lambda
                print('ESTIMATED RATE: {:.6f}, MET: {:.2f}min'.format(est_lambda, MET))
            else:
                est_lambda = initial_guess
            est_lambdas.append(est_lambda)
        except ValueError:
            print('insufficient bursts ({} of {})'.format(i, len(b_e)))
            est_lambdas.append(initial_guess)

    t = [t * t_poll for (t, l) in enumerate(est_lambdas)]

    plt.figure(figsize=(10, 5))
    plt.plot(t, est_lambdas)

    avg_lambdas = []
    for i in range(len(est_lambdas)):
        try:
            avg = sum(est_lambdas[:i]) / i
            avg_lambdas.append(avg)
        except ZeroDivisionError:
            avg_lambdas.append(initial_guess)
    plt.plot(t, avg_lambdas, 'r')
    plt.show()

    F = [1 - math.exp(-l * t * t_poll) for (t, l) in enumerate(est_lambdas)]
    F_avg = [1 - math.exp(-l * t * t_poll) for (t, l) in enumerate(avg_lambdas)]
    F_init = [1 - math.exp(-initial_guess * t * t_poll) for (t, l) in enumerate(avg_lambdas)]
    F_post = [1 - math.exp(-avg_lambdas[-1] * t * t_poll) for (t, l) in enumerate(avg_lambdas)]
    print(F_post)
    plt.figure(figsize=(10, 5))
    # plt.plot(t, F, label='instantaneous l')
    plt.plot(t, F_avg, 'r', label='avg. l')
    plt.plot(t, F_init, 'g', label='initial guess')
    plt.plot(t, F_post, 'k', label='post. l')
    plt.plot()
    plt.legend()
    plt.show()
