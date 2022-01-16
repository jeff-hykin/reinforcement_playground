# from: https://raw.githubusercontent.com/PaoloLRinaldi/progress_bar_python/master/perc.py
from datetime import datetime, timedelta
import time
import sys

try:
    import numpy as np
except ImportError:
    np = None


class ProgressBar:
    def __init__(self, v_max, verbose=3, inline=True, show_bar=True, disable=False, title=None):
        if isinstance(v_max, int):
            self._v_max = v_max
            self._it = 0
        else:
            self._v_max = len(v_max)
            self._it = -1
            self._tomanage = iter(v_max)
        self._perc = -1
        self._times = [time.time()]
        self._verbose = verbose
        self._inline = inline
        self._show_bar = show_bar
        self._progsz = 20
        self._passedits = list()
        self._disable = disable
        self._title = title
        self._starttime = datetime.now()

    def __new__(cls, *args, **kwargs):
        return super(ProgressBar, cls).__new__(cls)

    def tomins(self, secs):
        secs = int(round(secs))
        mins = secs // 60
        secs = secs % 60
        strhours = ''
        if mins >= 60:
            hours = mins // 60
            mins = mins % 60
            mins = "{:02d}".format(mins)
            strhours = str(hours) + ':'

        if secs < 10:
            return strhours + '{}:0{}'.format(mins, secs)
        return strhours + '{}:{}'.format(mins, secs)

    def next(self, it=None):
        if self._disable:
            return
        if it is not None:
            self._it = it
        self._it += 1
        current_perc = self._it * 100 // self._v_max
        if current_perc != self._perc:
            if self._inline:
                print('\r', end='')
            if self._title is not None:
                print(self._title, end=' ')
            if self._show_bar:
                prog = int((self._it / self._v_max) * self._progsz)
                print('[' + '=' * prog, end='')
                if prog != self._progsz:
                    print('>' + '.' * (self._progsz - prog - 1), end='')
                print('] ', end='')
            print('{}%'.format(current_perc), end='')
            if self._verbose > 0:
                self._times.append(time.time())
                self._passedits.append(self._it)
                if len(self._times) > 2:
                    step = self._times[-1] - self._times[-2]
                    itspersec = (self._passedits[-1] - self._passedits[-2]) / step
                    print(' | %i/%i %smin/perc %.2fit/s' % (self._it, self._v_max, self.tomins(step), itspersec), end='')
                    if self._verbose > 1:
                        elps = self._times[-1] - self._times[0]
                        print(' | %s' % (self.tomins(elps)), end='')
                        if self._verbose > 2 and current_perc != 100 and np:
                            p = np.poly1d(
                                np.polyfit(self._passedits, self._times[1:], w=np.arange(1, len(self._times)), deg=1))
                            secs_remaining = p(self._v_max) - p(self._it)
                            print('<%s => %s' % (self.tomins(secs_remaining), self.tomins(secs_remaining + elps)),
                                  end='')
                            if self._verbose > 3:
                                endtime = self._starttime + timedelta(seconds=elps + secs_remaining)
                                print(' | Started: %s - Ends at: %s' % (
                                    self._starttime.strftime("%H:%M:%S"), endtime.strftime("%H:%M:%S")), end='')
                                if self._verbose > 4:
                                    nxt = p(int(round((current_perc + 1) * self._v_max / 100 - 0.5) + 1)) - p(self._it)
                                    print(' | Next in %.1f/%s' % (nxt, self.tomins(nxt)), end='')
                if not self._inline:
                    print()
            if self._it == self._v_max:
                self._printdone()
            self._perc = current_perc

    def _printdone(self):
        if self._inline:
            print('\r', end='')
            sys.stdout.write('\033[2K\033[1G')
        print('Done in %s at %s' % (self.tomins(time.time() - self._times[0]), datetime.now().strftime("%H:%M:%S")))

    def done(self):
        if self._it != self._v_max and not self._disable:
            self._printdone()

    def __iter__(self):
        return self

    def __next__(self):
        if self._it < self._v_max:
            try:
                self.next()
            except ZeroDivisionError:
                pass
            return next(self._tomanage)
        else:
            raise StopIteration


if __name__ == "__main__":
    for i in ProgressBar(range(100), verbose=1):
        time.sleep(0.1)