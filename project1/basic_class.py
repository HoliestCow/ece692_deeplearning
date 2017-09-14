import time
from .printers import print_info, print_warning, AnsiColors


class BasicClass(object):

    _tic_sec = time.time()

    @property
    def name(self):
        return str(type(self).__name__)

    def _print(self, *s):
        print_info(self.name, *s)

    def _warning(self, *s):
        print_warning(*s, title=self.name)

    def _tic(self):
        self._tic_sec = time.time()

    def _tprint(self, *s, tic=True):
        self._print(
            AnsiColors.magenta(
                '({:9.6f} sec)'.format(time.time() - self._tic_sec)),
            *s)
        if tic:
            self._tic()
