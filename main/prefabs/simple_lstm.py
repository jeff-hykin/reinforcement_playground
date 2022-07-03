import torch
import torch.nn as nn
from trivial_torch_tools import to_tensor

class SimpleLstm(nn.LSTM):
    """
    network = SimpleLstm(input_size=10, output_size=3, number_of_layers=2)

    sequence = [
        torch.randn(input_size), # timestep #1
        torch.randn(input_size), # timestep #2
        torch.randn(input_size), # timestep #3
        torch.randn(input_size), # timestep #4
        torch.randn(input_size), # timestep #5
    ]

    network.forward(sequence).shape # (5, 3) one output per element in the sequence

    pipeline = network.pipeline()
    pipeline(sequence[0]).shape     # (3,) singular output
    pipeline(sequence[1]).shape     # (3,) singular output (and LSTM state is preserved)
    pipeline(sequence[2]).shape     # (3,) singular output (and LSTM state is preserved)
    pipeline(sequence[3]).shape     # (3,) singular output (and LSTM state is preserved)
    pipeline(sequence[4]).shape     # (3,) singular output (and LSTM state is preserved)
    """
    
    
    def __init__(self, *, input_size, output_size, number_of_layers=2, **kwargs):
        self.input_size = input_size
        self.output_size = output_size
        self.number_of_layers = number_of_layers
        super(SimpleLstm, self).__init__(input_size, output_size, number_of_layers, batch_first=True, **kwargs)
    
    def forward(self, sequence_or_batch, hidden_init=None):
        sequence_or_batch = to_tensor(sequence_or_batch)
        sequence_length = len(sequence_or_batch)
        real_forward = super(SimpleLstm, self).forward
        # batch
        if len(sequence_or_batch.shape) == 3:
            output, (initial_hidden_states, cell_states) = real_forward(sequence_or_batch, hidden_init)
            return output
            
        # no batch
        else:
            batch_size = 1
            batched_input = sequence_or_batch.reshape((batch_size, sequence_length, self.input_size))
            if type(hidden_init) == type(None):
                outputs = real_forward(batched_input)
            else:
                h0 = hidden_init[0].reshape((batch_size, self.number_of_layers, self.output_size))
                c0 = hidden_init[1].reshape((batch_size, self.number_of_layers, self.output_size))
                outputs = real_forward(batched_input, (h0, c0))
            
            output, (initial_hidden_states, cell_states) = outputs
            return output.reshape((sequence_length, self.output_size))
    
    def forward_full(self, sequence_or_batch, hidden_init=None):
        sequence_or_batch = to_tensor(sequence_or_batch)
        real_forward = super(SimpleLstm, self).forward
        sequence_length = len(sequence_or_batch)
        # batch
        if len(sequence_or_batch.shape) == 3:
            return real_forward(sequence_or_batch, hidden_init)
        # no batch
        else:
            batch_size = 1
            batched_input = sequence_or_batch.reshape((batch_size, sequence_length, self.input_size))
            if type(hidden_init) == type(None):
                outputs = real_forward(batched_input)
            else:
                h0 = hidden_init[0].reshape((self.number_of_layers, batch_size, self.output_size))
                c0 = hidden_init[1].reshape((self.number_of_layers, batch_size, self.output_size))
                outputs = real_forward(batched_input, (h0, c0))
            
            output, (initial_hidden_states, cell_states) = outputs
            return (
                output.reshape((sequence_length, self.output_size)),
                (
                    initial_hidden_states.reshape((self.number_of_layers, self.output_size)),
                    cell_states.reshape((self.number_of_layers, self.output_size)),
                )
            )
        
    def pipeline(self):
        def process_next(input_frame):
            batch_size = 1
            sequence_size = 1
            output_sequence, process_next.previous_hidden_values = self.forward_full(
                input_frame.reshape((batch_size, sequence_size, *input_frame.shape)),
                process_next.previous_hidden_values,
            )
            process_next.previous_output = output_sequence[0][0]
            return process_next.previous_output
        
        process_next.previous_output = None
        process_next.previous_hidden_values = None
        
        return process_next


