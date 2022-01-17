# from: https://raw.githubusercontent.com/PaoloLRinaldi/progress_bar_python/master/perc.py
from datetime import datetime, timedelta
import time
import sys
import numpy as np
import math

class ProgressBar:
    """
    for each in ProgressBar(range(100)):
        pass
    """
    layout = [ 'remaining_time', 'spacer', 'bar', 'percent', 'spacer', 'fraction', 'spacer', 'start_time' ]
    def __init__(self, iterations, inline=True, show_bar=True, disable=False, title=None, progress_bar_size=20, update_frequency=10, layout=None):
        if isinstance(iterations, int):
            self.total_iterations = iterations
            self.iteration = 0
        else:
            self.total_iterations = len(iterations)
            self.iteration = -1
            self._iterator = iter(iterations)
        self.prev_percent = -1
        self._times = [time.time()]
        self._inline = inline
        self.should_show_bar = show_bar
        self.progress_bar_size = progress_bar_size
        self._past_iterations = list()
        self._disable = disable
        self.title = title
        self.start_time = datetime.now()
        self.update_frequency = update_frequency
        self.prev_time = -math.inf
        self.layout = layout

    def __new__(cls, *args, **kwargs):
        return super(ProgressBar, cls).__new__(cls)

    def to_time_string(self, secs):
        secs = int(round(secs))
        mins = secs // 60
        secs = secs % 60
        str_hours = ''
        if mins >= 60:
            hours = mins // 60
            mins = mins % 60
            mins = "{:02d}".format(mins)
            str_hours = str(hours) + ':'

        if secs < 10:
            return str_hours + '{}:0{}'.format(mins, secs)
        return str_hours + '{}:{}'.format(mins, secs)
    
    def next(self, it=None):
        if self._disable:
            return
        if it is not None:
            self.iteration = it
        
        self.iteration += 1
        self.current_percent = (self.iteration * 10000 // self.total_iterations) / 100
        now = time.time()
        if now - self.prev_time > self.update_frequency:
            self.prev_time = now
            # compute data
            self._times.append(time.time())
            self._past_iterations.append(self.iteration)
            self.total_eslaped_time = 0
            self.eslaped_time       = 0
            self.secs_remaining     = math.inf
            # compute changes (need at least two times to do that)
            if len(self._times) > 2:
                self.total_eslaped_time = self._times[-1] - self._times[ 0]
                self.eslaped_time       = self._times[-1] - self._times[-2]
                p = np.poly1d(
                    np.polyfit(
                        self._past_iterations,
                        self._times[1:],
                        w=np.arange(1, len(self._times)),
                        deg=1
                    )
                )
                self.secs_remaining = p(self.total_iterations) - p(self.iteration)
            
            if self._inline:
                print('\r', end='')
            
            if self.title is not None:
                print(self.title, end=' ')
            
            # display each thing according to the layout
            for each in (self.layout or ProgressBar.layout):
                getattr(self, f"show_{each}", lambda : None)()
            
            # padding
            if not self._inline:
                print()
            else:
                self.show_spacer()
            
            if self.iteration == self.total_iterations:
                self.show_done()
            
            self.prev_percent = self.current_percent

    def show_spacer(self):
        print(" | ", end='')
        
    def show_bar(self):
        if self.should_show_bar:
            prog = int((self.iteration / self.total_iterations) * self.progress_bar_size)
            print('[' + '=' * prog, end='')
            if prog != self.progress_bar_size:
                print('>' + '.' * (self.progress_bar_size - prog - 1), end='')
            print('] ', end='')
    
    def show_remaining_time(self):
        if self.secs_remaining == math.inf:
            print(f'remaining: ________', end='')
        elif self.current_percent != 100:
            print(f'remaining: {self.to_time_string(self.secs_remaining)}sec', end='')
        
    def show_percent(self):
        print('{}%'.format(self.current_percent), end='')
    
    def show_duration(self):
        print('%s' % (self.to_time_string(self.total_eslaped_time)), end='')
    
    def show_fraction(self):
        print(f'{self.iteration}/{self.total_iterations}', end='')
        
    def show_iteration_time(self):
        iterations_per_sec = (self._past_iterations[-1] - self._past_iterations[-2]) / self.eslaped_time
        print(f'{self.to_time_string(self.eslaped_time)}sec per iter', end='')
    
    def show_start_time(self):
        if self.current_percent != 100:
            print(f'started: {self.start_time.strftime("%H:%M:%S")}',  end='')
    
    def show_end_time(self):
        endtime = self.start_time + timedelta(seconds=self.total_eslaped_time + self.secs_remaining)
        if self.current_percent != 100:
            print(f'eta: {endtime.strftime("%H:%M:%S")}',  end='')
    
    def show_done(self):
        if self._inline:
            print('\r', end='')
            sys.stdout.write('\033[2K\033[1G')
        print('Done in %s at %s' % (self.to_time_string(time.time() - self._times[0]), datetime.now().strftime("%H:%M:%S")))

    def done(self):
        if self.iteration != self.total_iterations and not self._disable:
            self.show_done()

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration < self.total_iterations:
            try:
                self.next()
            except ZeroDivisionError:
                pass
            return next(self._iterator)
        else:
            raise StopIteration


if __name__ == "__main__":
    for i in ProgressBar(range(100)):
        time.sleep(0.1)