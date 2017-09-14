import os
import sys
import multiprocessing
import copyreg
import types
import time
from six import string_types
from progressbar import ProgressBar, FormatLabel, Percentage, Bar, ETA
import numpy as np
import pandas as pd
from subprocess import Popen
from collections import OrderedDict
from basic_class import BasicClass


MAX_NCORES = multiprocessing.cpu_count()
SAFE_NCORES = MAX_NCORES - 2


# -----------------------------------------------------------------------------
# This is a trick to allow multiprocessing to use target functions that are
# object methods. This is used for the algorithms which are trained and then
# evaluations are completed inside of MP threads
# -----------------------------------------------------------------------------
def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)


copyreg.pickle(types.MethodType, _pickle_method)


def _mute_stdout():
    sys.stdout = open(os.devnull, 'w')


class _MultiProcessor(BasicClass):

    def __init__(self, ncores=SAFE_NCORES):
        self.ncores = ncores

    @property
    def ncores(self):
        return self._ncores

    @ncores.setter
    def ncores(self, x):
        if x is None:
            ncores = self._ncores
        elif isinstance(x, string_types):
            if x.lower() == 'max':
                ncores = MAX_NCORES
            elif x.lower() == 'safe':
                ncores = SAFE_NCORES
            elif x.isdigit():
                ncores = int(x)
            else:
                raise ValueError('Unrecognized `ncores`: {}'.format(x))
        else:
            ncores = int(x)
        if ncores <= 0:
            raise ValueError('`ncores` must be positive: {}'.format(ncores))
        if ncores > MAX_NCORES:
            raise ValueError(
                'ncores={} exceeds MAX_NCORES={}'.format(ncores, MAX_NCORES))
        self._ncores = ncores
        self._print('Using {} cores'.format(self.ncores))


class PyMultiProcessor(_MultiProcessor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        copyreg.pickle(types.MethodType, _pickle_method)
        self.reset_cmds()

    def reset_cmds(self):
        self.funcs = []
        self.args = []
        self.num_cmds = 0

    def add_func_and_args(self, func, args):
        """
        Add one function and dict of args to be processed
        """
        assert isinstance(args, dict), '`args` not dict: {}'.format(args)
        self.funcs.append(func)
        self.args.append(args)
        self.num_cmds += 1

    def run_processes(self, ncores=None, concat=False,
                      mute_stdout=True):
        """
        Run all functions and args added via `add_func_and_args`
        If `ncores=1` multiprocessing is not used.
        If `concat` the results will be attempted to be concatenated along
        axis=1 (column-wise)
        """
        self.ncores = ncores
        # ProgressBar stuff
        widgets = [
            FormatLabel('Processed: %(value)d of {} '.format(self.num_cmds)),
            Percentage(),
            Bar(),
            ETA()]
        # Storage
        results = []
        # Single process
        if self.ncores == 1:
            pbar = ProgressBar(widgets=widgets, maxval=self.num_cmds).start()
            for i, (f, a) in enumerate(zip(self.funcs, self.args)):
                results.append(f(**a))
                pbar.update(i + 1)
            pbar.finish()
        # Multiprocess
        else:
            pbar = ProgressBar(widgets=widgets, maxval=self.num_cmds).start()
            if mute_stdout:
                self.pool = multiprocessing.Pool(processes=self.ncores,
                                                 initializer=_mute_stdout)
            else:
                self.pool = multiprocessing.Pool(processes=self.ncores)
            procs = []
            # Start procs
            for i, (f, a) in enumerate(zip(self.funcs, self.args)):
                procs.append(self.pool.apply_async(f, (), a))
            # Wait for and collect results
            for i, p in enumerate(procs):
                results.append(p.get())
                pbar.update(i + 1)
            pbar.finish()
            self.pool.close()
        if concat:
            # Concat dataframes?
            if all([isinstance(x, pd.DataFrame) for x in results]):
                self.results = pd.concat(results, axis=0)
                self.results.sort_index(inplace=True)
            # Concat dicts of arrays?
            elif all([isinstance(x, dict) for x in results]):
                self.results = OrderedDict()
                # Commented this to make it work.
                # for result in results:
                #     for k in result:
                #         if not isinstance(result[k], np.ndarray):
                #             result[k] = np.array(result[k])
                #         append_to_dict_list(self.results, k, result[k])
                for k in self.results:
                    self.results[k] = np.concatenate(results[k], axis=0)
            else:
                self.results = results
        else:
            self.results = results
        # Cleanup
        self.reset_cmds()
        return self.results


def run_process(cmd=[], log_fname=None, **kwargs):
    """
    cmd_dict is a dictionary of the command line command and arguments,
    log filename and the process number (ID)
    """
    # Starting time
    start_time = time.time()
    # Discard stdout to /dev/null
    if (log_fname == 'null') or (log_fname is False):
        proc = Popen(cmd, stdout=open(os.devnull, 'w'))
    # Print stdout normally
    elif log_fname is None:
        proc = Popen(cmd)
    # Save stdout to file
    else:
        proc = Popen(cmd, stdout=open(log_fname, 'w'))
    # Wait for command to finish...
    proc.wait()
    end_time = time.time()
    return {'pid': proc.pid,
            'cmd': ' '.join(cmd),
            'start_time': start_time,
            'end_time': end_time,
            'duration_sec': end_time - start_time}
