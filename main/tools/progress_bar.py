# from: https://raw.githubusercontent.com/PaoloLRinaldi/progress_bar_python/master/perc.py
from datetime import datetime, timedelta
import time
import sys
from statistics import mean, stdev
import math
from super_map import Map

try:
    from IPython.display import display, HTML, clear_output
    from io import StringIO
    ipython_exists = True
except:
    ipython_exists = False
    
class NotGiven: pass

def subsequence_replace(a_list, sequence, replacement):
    que = []
    new_list = []
    for each in a_list:
        que.append(each)
        while len(que) > len(sequence):
            new_list.append(que.pop(0))
        if que == sequence:
            que.clear()
            for each in replacement:
                new_list.append(each)
    new_list += que # add any remaining elements
    return new_list    

class ProgressBar:
    """
    from tools.progress_bar import ProgressBar, time
    for progress, each in ProgressBar(10000):
        time.sleep(0.01)
    """
    layout = [ 'title', 'bar', 'percent', 'spacer', 'fraction', 'spacer', 'start_time', 'spacer', 'end_time', 'spacer', 'remaining_time', 'spacer', ]
    minimal_layout = [ 'title', 'bar', 'spacer', 'end_time', 'spacer', ]
    spacer = " | "
    minmal = False
    inline = True
    disable_print = False
    progress_bar_size = 35
    seconds_per_print = 1
    percent_per_print = 10
    
    @classmethod
    def configure(this_class, **config):
        for each_key, each_value in config.items():
            setattr(this_class, each_key, each_value)
    
    def __init__(self, iterations, *, title=None, layout=None, disable_print=NotGiven, minimal=NotGiven, inline=NotGiven, progress_bar_size=NotGiven, seconds_per_print=NotGiven, percent_per_print=NotGiven, ):
        original_generator = range(int(iterations)) if isinstance(iterations, (int, float)) else iterations
        self.title = title
        
        # inherit unspecified options from class object
        for each_option in [ "disable_print", "minimal", "inline", "progress_bar_size", "seconds_per_print", "percent_per_print", ]:
            arg_value = eval(each_option, locals())
            # default to the class value if not given
            if arg_value == NotGiven:
                actual_value = getattr(ProgressBar, each_option, None)
            # otherwise use the given value
            else:
                actual_value = arg_value
            # set the object's value
            setattr(self, each_option, actual_value)
        
        # initilize misc values
        self.past_indicies     = []
        self.start_time        = datetime.now()
        self.next_percent_mark = self.percent_per_print
        self.prev_time         = -math.inf
        self.times             = [time.time()]
        self.progress_data     = Map(
            index=0,
            percent=0,
            updated=True,
            time=self.times[0],
            total=len(original_generator),
        )
        # setup print
        if self.disable_print:
            self.print = lambda *args, **kwargs: None
        elif not ipython_exists:
            self.print = print
        else:
            # remove the progress bar and percent
            layout = list(self.layout)
            layout = subsequence_replace(layout, [ 'spacer', 'bar'    , 'spacer' ], ['spacer'])
            layout = subsequence_replace(layout, [ 'spacer', 'bar'    ,          ], ['spacer'])
            layout = subsequence_replace(layout, [           'bar'    , 'spacer' ], ['spacer'])
            layout = subsequence_replace(layout, [           'bar'               ], [])
            layout = subsequence_replace(layout, [ 'spacer', 'percent', 'spacer' ], ['spacer'])
            layout = subsequence_replace(layout, [ 'spacer', 'percent',          ], ['spacer'])
            layout = subsequence_replace(layout, [           'percent', 'spacer' ], ['spacer'])
            layout = subsequence_replace(layout, [           'percent',          ], [])
            layout = subsequence_replace(layout, [ 'spacer', 'spacer'            ], ['spacer'])
            self.layout = layout
            self.string_buffer = ""
            def ipython_print(*args, **kwargs):
                # get the string value
                string_stream = StringIO()
                print(*args, **kwargs, file=string_stream)
                output_str = string_stream.getvalue()
                string_stream.close()
                self.string_buffer += output_str
                
                # clear output whenever newline is created
                if kwargs.get("end", "\n") in ['\n', '\r']:
                    clear_output(wait=True)
                    display(HTML(f'''
                        <div style="width: 100%; background: transparent; color: white; position: relative; padding: 1.2rem 0.4rem; box-sizing: border-box;">
                            <div style="position: relative; background: #46505a; height: 2.7rem; width: 100%; border-radius: 10rem; border: transparent solid 0.34rem; box-sizing: border-box;">
                                <!-- color bar -->
                                <div style="height: 100%; width: {self.progress_data.percent}%; background: #9b68ab; border-radius: 10rem;"></div>
                                <!-- percentage text -->
                                <div style="height: 100%; width: 100%; position: absolute; top: 0; left: 0; display: flex; flex-direction: column; align-items: center; align-content: center; justify-items: center;  justify-content: center;">
                                    <span>
                                        {self.progress_data.percent:0.2f}%
                                    </span>
                                </div>
                            </div>
                            <div style="width: 100%; height: 1rem;">
                            </div>
                            <div style="position: relative; height: fit-content; width: 100%; box-sizing: border-box; display: flex; flex-direction: column; align-items: center; align-content: center; justify-items: center;  justify-content: center;">
                                <div style="position: relative; background: #46505a; height: 1.7rem; width: fit-content; border-radius: 10rem; border: transparent solid 0.5rem; box-sizing: border-box; display: flex; flex-direction: column; align-items: center; align-content: center; justify-items: center;  justify-content: center;">
                                    <code style="whitespace: pre; color: whitesmoke;" >
                                        {self.string_buffer}
                                    </code>
                                </div>
                            </div>
                        </div>
                    '''))
                    # clear the buffer
                    self.string_buffer = ""
            self.print = ipython_print
            
        # setup layout
        if layout == None and self.minimal:
            self.layout = ProgressBar.minimal_layout
        elif layout == None:
            self.layout = ProgressBar.layout
        else:
            self.layout = layout
        
        def generator_func():
            for self.progress_data.index, each_original in enumerate(original_generator):
                # update
                self.progress_data.time    = time.time()
                self.progress_data.updated = (self.progress_data.time - self.prev_time) > self.seconds_per_print
                self.progress_data.percent = (self.progress_data.index * 10000 // self.progress_data.total) / 100 # two decimals of accuracy
                
                if self.progress_data.updated:
                    self.prev_time = self.progress_data.time
                    self.times.append(self.progress_data.time)
                    self.past_indicies.append(self.progress_data.index)
                    
                # also printout at each percent marker
                if self.progress_data.percent >= self.next_percent_mark:
                    self.next_percent_mark += self.percent_per_print
                    self.progress_data.updated = True
                
                if self.progress_data.updated:
                    self.total_eslaped_time = 0
                    self.eslaped_time       = 0
                    self.secs_remaining     = math.inf
                    # compute changes (need at least two times to do that)
                    if len(self.times) > 3:
                        self.total_eslaped_time = self.times[-1] - self.times[ 0]
                        self.eslaped_time       = self.times[-1] - self.times[-2]
                        
                        # compute ETA as a slight overestimate that is less of an overesitmate over time
                        iterations_per_update = tuple(each-prev for prev, each in zip(self.past_indicies[0:-1], self.past_indicies[1:]))
                        average_iterations = mean(iterations_per_update)
                        deviation = stdev(iterations_per_update)
                        partial_deviation = (self.progress_data.percent/100) * deviation
                        lowerbound_iterations_per_update = average_iterations - partial_deviation
                        expected_number_of_updates_needed = (self.progress_data.total - self.progress_data.index) / lowerbound_iterations_per_update
                        time_per_update = self.total_eslaped_time / len(iterations_per_update)
                        self.secs_remaining = time_per_update * expected_number_of_updates_needed
                    
                    if self.inline:
                        self.print('', end='\r')
                    
                    # display each thing according to the layout
                    for each in self.layout:
                        getattr(self, f"show_{each}", lambda : None)()
                    
                    if not self.inline:
                        self.print()
                    
        
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
        self.print(self.spacer, end='')
    
    def show_title(self):
        if self.title is not None:
            self.print(self.title, end=' ')
        
    def show_bar(self):
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
        self.print(f'{self.progress_data.percent:.2f}%'.rjust(6), end='')
    
    def show_duration(self):
        self.print(self.to_time_string(self.total_eslaped_time), end='')
    
    def show_fraction(self):
        total_str = f"{self.progress_data.total}"
        self.print(f'{self.progress_data.index}'.rjust(len(total_str))+f'/{self.progress_data.total}', end='')
        
    def show_iteration_time(self):
        iterations_per_sec = (self.past_indicies[-1] - self.past_indicies[-2]) / self.eslaped_time
        self.print(f'{self.to_time_string(self.eslaped_time)}sec per iter', end='')
    
    def show_start_time(self):
        if self.progress_data.percent != 100:
            self.print(f'started: {self.start_time.strftime("%H:%M:%S")}',  end='')
    
    def show_end_time(self):
        if self.progress_data.percent != 100:
            time_format = "%H:%M:%S"
            try:
                endtime = self.start_time + timedelta(seconds=self.total_eslaped_time + self.secs_remaining)
                self.print(f'eta: {endtime.strftime("%H:%M:%S")}',  end='')
            except:
                self.print(f'eta: {"_"*len(time_format)}',  end='')
    
    def show_done(self):
        if self.inline:
            print("")
        duration = self.to_time_string(time.time() - self.times[0])
        end_time = datetime.now().strftime("%H:%M:%S")
        self.progress_data.percent = 100.0
        self.string_buffer = "" # for ipython
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