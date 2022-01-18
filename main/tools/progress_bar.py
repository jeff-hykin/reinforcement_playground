# from: https://raw.githubusercontent.com/PaoloLRinaldi/progress_bar_python/master/perc.py
from datetime import datetime, timedelta
import time
import sys
import numpy as np
import math
from super_map import Map
    
class ProgressBar:
    """
    for progress, each in ProgressBar(range(100)):
        pass
    """
    layout = [ 'remaining_time', 'spacer', 'bar', 'percent', 'spacer', 'fraction', 'spacer', 'start_time' ]
    
    def __init__(self, iterations, inline=True, show_bar=True, disable=False, title=None, progress_bar_size=25, update_frequency=10, layout=None):
        original_generator = range(int(iterations)) if isinstance(iterations, (int, float)) else iterations
        self.print = print if not disable else lambda *args, **kwargs: None
        self.times = [time.time()]
        self.inline            = inline
        self.should_show_bar   = show_bar
        self.progress_bar_size = progress_bar_size
        self.past_indicies     = list()
        self.title             = title
        self.start_time        = datetime.now()
        self.update_frequency  = update_frequency
        self.prev_time         = -math.inf
        self.layout            = layout
        self.progress_data = Map(
            index=0,
            percent=0,
            updated=True,
            time=self.times[0],
            total=len(original_generator),
        )
        def generator_func():
            for self.progress_data.index, each_original in enumerate(original_generator):
                self.progress_data.time    = time.time()
                self.progress_data.updated = self.progress_data.time - self.prev_time > self.update_frequency
                self.progress_data.percent = (self.progress_data.index * 10000 // self.progress_data.total) / 100
                a_ten_percent_mark = int(self.progress_data.percent % 10) == 0
                if self.progress_data.updated or a_ten_percent_mark:
                    self.prev_time = self.progress_data.time
                    # compute data
                    self.times.append(time.time())
                    self.past_indicies.append(self.progress_data.index)
                    self.total_eslaped_time = 0
                    self.eslaped_time       = 0
                    self.secs_remaining     = math.inf
                    # compute changes (need at least two times to do that)
                    if len(self.times) > 2:
                        self.total_eslaped_time = self.times[-1] - self.times[ 0]
                        self.eslaped_time       = self.times[-1] - self.times[-2]
                        p = np.poly1d(
                            np.polyfit(
                                self.past_indicies,
                                self.times[1:],
                                w=np.arange(1, len(self.times)),
                                deg=1
                            )
                        )
                        self.secs_remaining = p(self.progress_data.total) - p(self.progress_data.index)
                    
                    if self.inline:
                        self.print('\r', end='')
                    
                    if self.title is not None:
                        self.print(self.title, end=' ')
                    
                    # display each thing according to the layout
                    for each in (self.layout or ProgressBar.layout):
                        getattr(self, f"show_{each}", lambda : None)()
                    
                    # padding
                    if not self.inline:
                        self.print()
                    else:
                        self.show_spacer()
                    
                    if self.progress_data.index == self.progress_data.total:
                        self.show_done()
                    
        
                yield self.progress_data, each_original
            
            self.show_done()
            
        self.iterator = iter(generator_func())

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
    
    def show_spacer(self):
        self.print(" | ", end='')
        
    def show_bar(self):
        if self.should_show_bar:
            prog = int((self.progress_data.index / self.progress_data.total) * self.progress_bar_size)
            self.print('[' + '=' * prog, end='')
            if prog != self.progress_bar_size:
                self.print('>' + '.' * (self.progress_bar_size - prog - 1), end='')
            self.print('] ', end='')
    
    def show_remaining_time(self):
        if self.secs_remaining == math.inf:
            self.print(f'remaining: ________', end='')
        elif self.progress_data.percent != 100:
            self.print(f'remaining: {self.to_time_string(self.secs_remaining)}sec', end='')
        
    def show_percent(self):
        self.print(f'{self.progress_data.percent:.2f}%', end='')
    
    def show_duration(self):
        self.print(self.to_time_string(self.total_eslaped_time), end='')
    
    def show_fraction(self):
        self.print(f'{self.progress_data.index}/{self.progress_data.total}', end='')
        
    def show_iteration_time(self):
        iterations_per_sec = (self.past_indicies[-1] - self.past_indicies[-2]) / self.eslaped_time
        self.print(f'{self.to_time_string(self.eslaped_time)}sec per iter', end='')
    
    def show_start_time(self):
        if self.progress_data.percent != 100:
            self.print(f'started: {self.start_time.strftime("%H:%M:%S")}',  end='')
    
    def show_end_time(self):
        endtime = self.start_time + timedelta(seconds=self.total_eslaped_time + self.secs_remaining)
        if self.progress_data.percent != 100:
            self.print(f'eta: {endtime.strftime("%H:%M:%S")}',  end='')
    
    def show_done(self):
        if self.inline:
            print("")
        duration = self.to_time_string(time.time() - self.times[0])
        end_time = datetime.now().strftime("%H:%M:%S")
        self.print(f'Done in {duration}sec at {end_time}')

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.iterator)
    
    def __len__(self):
        return self.progress_data.total


if __name__ == "__main__":
    for progress, each in ProgressBar(range(100)):
        time.sleep(0.1)