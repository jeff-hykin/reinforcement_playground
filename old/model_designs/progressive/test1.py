import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim

from agents.progressive.parts import *

torch.manual_seed(111223)   # Uncomment to make deterministic.


class LogicGen(ProgColumnGenerator):
    def __init__(self):
        self.ids = 0

    def generateColumn(self, parentCols, msg = None):
        b1 = ProgDenseBlock(2, 3, 0, activation = nn.ReLU())
        b2 = ProgDenseBlock(3, 1, len(parentCols), activation = nn.Sigmoid())
        c = ProgColumn(self.__genID(), [b1, b2], parentCols = parentCols)
        return c

    def __genID(self):
        id = self.ids
        self.ids += 1
        return id

def main():
    # 
    # 
    # XOR
    # 
    # 
    if True:
        # 
        # prep work
        # 
        net        = ProgNet(colGen=LogicGen())
        idXor      = net.addColumn()
        inputsXor  = list(map(lambda s: Variable(torch.Tensor([s])), [[0 , 0] , [0 , 1], [1, 0], [1, 1]]))
        targetsXor = list(map(lambda s: Variable(torch.Tensor([s])), [[0], [1], [1], [0]]))
        mse        = nn.MSELoss()
        optimizer  = optim.SGD(net.parameters(), lr = 0.001)
        
        # 
        # training
        # 
        print("Training Xor.")
        for i in range(25000):
            for input, target in zip(inputsXor, targetsXor):
                optimizer.zero_grad()
                output = net(idXor, input)
                loss = mse(output, target)
                loss.backward()
                optimizer.step()
            if i % 2500 == 0:
                print("   Epoch %d. Loss: %f." % (i, loss))
        
        # 
        # testing
        # 
        print("Testing Xor.")
        for input, target in zip(inputsXor, targetsXor):
            output = net(idXor, input)
            in0 = int(input.data.numpy()[0][0])
            in1 = int(input.data.numpy()[0][1])
            targ = int(target.data.numpy()[0])
            pred = round(float(output.data.numpy()[0]))
            err = abs(targ - pred)
            print("   Input: [%d, %d].  Target: %d.  Predicted: %d.  Error: %d." % (in0, in1, targ, pred, err))
        
        # 
        # parameters
        # 
        print("Xor params:")
        for n, p in net.named_parameters():
            print(n)
            print(p)
        print("\n")
        
    
    # 
    # 
    # Nand
    # 
    # 
    if True:
        # 
        # prep work
        # 
        net.freezeAllColumns()
        idNand      = net.addColumn()
        inputsNand  = list(map(lambda s: Variable(torch.Tensor([s])), [[0 , 0] , [0 , 1], [1, 0], [1, 1]]))
        targetsNand = list(map(lambda s: Variable(torch.Tensor([s])), [[1], [1], [1], [0]]))
        optimizer   = optim.SGD(net.parameters(), lr=0.01)
        
        # 
        # train
        # 
        print("Training Nand.")
        for i in range(25000):
            for input, target in zip(inputsNand, targetsNand):
                optimizer.zero_grad()
                output = net(idNand, input)
                loss = mse(output, target)
                loss.backward()
                optimizer.step()
            if i % 2500 == 0:
                print("   Epoch %d. Loss: %f." % (i, loss))
        
        # 
        # test
        # 
        print("Testing Nand.")
        for input, target in zip(inputsNand, targetsNand):
            output = net(idNand, input)
            in0 = int(input.data.numpy()[0][0])
            in1 = int(input.data.numpy()[0][1])
            targ = int(target.data.numpy()[0])
            pred = round(float(output.data.numpy()[0]))
            err = abs(targ - pred)
            print("   Input: [%d, %d].  Target: %d.  Predicted: %d.  Error: %d." % (in0, in1, targ, pred, err))
        
        # 
        # parameters
        # 
        print("Xor and Nand params:")
        for n, p in net.named_parameters():
            print(n)
            print(p)

main()