import tools.pytorch_tools as pytorch_tools
from slick_siphon import siphon
import torch

class FrameQue:
    def __init__(self, *, que_size, frame_shape):
        self.tensor = torch.zeros((que_size, *frame_shape))
    
    def add(self, latest_frame):
        # shift all the frames down by one (starting at the end and working backwards)
        for frame_index, each_frame in reversed(tuple(enumerate(self.tensor[:-1]))):
            self.tensor[frame_index+1] = self.tensor[frame_index]
        # push in the newest frame
        self.tensor[0] = pytorch_tools.to_tensor(latest_frame)
        return self.tensor
    
    @property
    def shape(self,):
        return self.tensor.shape
    
    # add to_tensor support
    @siphon(  when=(lambda arg1, *args, **kwargs: isinstance(arg1, (FrameQue,))),  is_true_for=pytorch_tools.to_tensor   )
    def to_tensor_extension(*args, **kwargs):
        return args[0].tensor
