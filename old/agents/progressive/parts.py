from tools.all_tools import *

"""
Class that acts as the base building-blocks of ProgNets.
Includes a module (usually a single layer),
a set of lateral modules, and an activation.
"""
class ProgBlock(nn.Module):
    """
    Runs the block on input x.
    Returns output tensor or list of output tensors.
    """
    def runBlock(self, x):
        raise NotImplementedError

    """
    Runs lateral i on input x.
    Returns output tensor or list of output tensors.
    """
    def runLateral(self, i, x):
        raise NotImplementedError

    """
    Runs activation of the block on x.
    Returns output tensor or list of output tensors.
    """
    def runActivation(self, x):
        raise NotImplementedError



"""
A ProgBlock containing a single fully connected layer (nn.Linear).
Activation function can be customized but defaults to nn.ReLU.
"""
class ProgDenseBlock(ProgBlock):
    def __init__(self, inSize, outSize, numLaterals, activation = nn.ReLU()):
        super().__init__()
        self.numLaterals = numLaterals
        self.inSize = inSize
        self.outSize = outSize
        self.module = nn.Linear(inSize, outSize)
        self.laterals = nn.ModuleList([nn.Linear(inSize, outSize) for _ in range(numLaterals)])
        if activation is None:   self.activation = (lambda x: x)
        else:                    self.activation = activation

    def runBlock(self, x):
        return self.module(x)

    def runLateral(self, i, x):
        lat = self.laterals[i]
        return lat(x)

    def runActivation(self, x):
        return self.activation(x)
    
    def isLateralized(self):
        return True


"""
A ProgBlock containing a single fully connected layer (nn.Linear) and a batch norm.
Activation function can be customized but defaults to nn.ReLU.
"""
class ProgDenseBNBlock(ProgBlock):
    def __init__(self, inSize, outSize, numLaterals, activation = nn.ReLU(), bnArgs = dict()):
        super().__init__()
        self.numLaterals = numLaterals
        self.inSize = inSize
        self.outSize = outSize
        self.module = nn.Linear(inSize, outSize)
        self.moduleBN = nn.BatchNorm1d(outSize, **bnArgs)
        self.laterals = nn.ModuleList([nn.Linear(inSize, outSize) for _ in range(numLaterals)])
        self.lateralBNs = nn.ModuleList([nn.BatchNorm1d(outSize, **bnArgs) for _ in range(numLaterals)])
        if activation is None:   self.activation = (lambda x: x)
        else:                    self.activation = activation

    def runBlock(self, x):
        return self.moduleBN(self.module(x))

    def runLateral(self, i, x):
        lat = self.laterals[i]
        bn = self.lateralBNs[i]
        return bn(lat(x))

    def runActivation(self, x):
        return self.activation(x)



"""
A ProgBlock containing a single Conv2D layer (nn.Conv2d).
Activation function can be customized but defaults to nn.ReLU.
Stride, padding, dilation, groups, bias, and padding_mode can be set with layerArgs.
"""
class ProgConv2DBlock(ProgBlock):
    def __init__(self, inSize, outSize, kernelSize, numLaterals, activation = nn.ReLU(), layerArgs = dict()):
        super().__init__()
        self.numLaterals = numLaterals
        self.inSize = inSize
        self.outSize = outSize
        self.kernSize = kernelSize
        self.module = nn.Conv2d(inSize, outSize, kernelSize, **layerArgs)
        self.laterals = nn.ModuleList([nn.Conv2d(inSize, outSize, kernelSize, **layerArgs) for _ in range(numLaterals)])
        if activation is None:   self.activation = (lambda x: x)
        else:                    self.activation = activation

    def runBlock(self, x):
        return self.module(x)

    def runLateral(self, i, x):
        lat = self.laterals[i]
        return lat(x)

    def runActivation(self, x):
        return self.activation(x)



"""
A column representing one sequential ANN with all of its lateral modules.
Outputs of the last forward run are stored for child column laterals.
Output of each layer is calculated as:
y = activation(block(x) + sum(laterals(x)))
"""
class ProgColumn(nn.Module):
    def __init__(self, colID, blockList, parentCols = []):
        super().__init__()
        self.colID = colID
        self.isFrozen = False
        self.parentCols = parentCols
        self.blocks = nn.ModuleList(blockList)
        self.numRows = len(blockList)
        self.lastOutputList = []

    def freeze(self, unfreeze = False):
        if not unfreeze:    # Freeze params.
            self.isFrozen = True
            for param in self.parameters():   param.requires_grad = False
        else:               # Unfreeze params.
            self.isFrozen = False
            for param in self.parameters():   param.requires_grad = True

    def forward(self, input):
        outputs = []
        x = input
        for row, block in enumerate(self.blocks):
            currOutput = block.runBlock(x)
            if row == 0 or len(self.parentCols) < 1:
                y = block.runActivation(currOutput)
            else:
                for c, col in enumerate(self.parentCols):
                    currOutput += block.runLateral(c, col.lastOutputList[row - 1])
                y = block.runActivation(currOutput)
            outputs.append(y)
            x = y
        self.lastOutputList = outputs
        return outputs[-1]



"""
A progressive neural network as described in Progressive Neural Networks (Rusu et al.).
Columns can be added manually or with a ProgColumnGenerator.
https://arxiv.org/abs/1606.04671
"""
class ProgNet(nn.Module):
    def __init__(self, colGen = None):
        super().__init__()
        self.columns = nn.ModuleList()
        self.numRows = None
        self.numCols = 0
        self.colMap = dict()
        self.colGen = colGen
        self.colShape = None

    def addColumn(self, col = None, msg = None):
        if not col:
            parents = [colRef for colRef in self.columns]
            col = self.colGen.generateColumn(parents, msg)
        self.columns.append(col)
        self.colMap[col.colID] = self.numCols
        self.numRows = col.numRows
        self.numCols += 1
        return col.colID

    def freezeColumn(self, id):
        col = self.columns[self.colMap[id]]
        col.freeze()

    def freezeAllColumns(self):
        for col in self.columns:
            col.freeze()

    def unfreezeColumn(self, id):
        col = self.columns[self.colMap[id]]
        col.freeze(unfreeze = True)

    def unfreezeAllColumns(self):
        for col in self.columns:
            col.freeze(unfreeze = True)

    def getColumn(self, id):
        col = self.columns[self.colMap[id]]
        return col

    def forward(self, id, x):
        colToOutput = self.colMap[id]
        for i, col in enumerate(self.columns):
            y = col(x)
            if i == colToOutput:
                return y


"""
Class that generates new ProgColumns using the method generateColumn.
The parentCols list will contain references to each parent column,
such that columns can access lateral outputs.
Additional information may be passed through the msg argument in
generateColumn and ProgNet.addColumn.
"""
class ProgColumnGenerator:
    def generateColumn(self, parentCols, msg = None):
        raise NotImplementedError
        
        


"""
A special case of ProgBlock with multiple paths.
"""
class ProgMultiBlock(ProgBlock):
    """
    Returns a list of booleans (pass_list).
    Length of the pass_list is equal to the number of channels in the block.
    Channels that return True do not operate on their inputs, and simply pass them to the next block.
    """
    def getPassDescriptor(self):
        raise NotImplementedError


class ProgBlock(nn.Module):
    def runBlock(self, x):
        raise NotImplementedError
        
    def runLateral(self, i, x):
        raise NotImplementedError

    def runActivation(self, x):
        raise NotImplementedError
        
    def isLateralized(self):
        return True


class ProgInertBlock(ProgBlock):
    def isLateralized(self):
        return False


class ProgColumn(nn.Module):
    
    def __init__(self, colID, blockList, parentCols = []):
        super().__init__()
        self.colID = colID
        self.isFrozen = False
        self.parentCols = parentCols
        self.blocks = nn.ModuleList(blockList)
        self.numRows = len(blockList)
        self.lastOutputList = []

    def freeze(self, unfreeze = False):
        if not unfreeze:    # Freeze params.
            self.isFrozen = True
            for param in self.parameters():   param.requires_grad = False
        else:               # Unfreeze params.
            self.isFrozen = False
            for param in self.parameters():   param.requires_grad = True

    def forward(self, input):
        outputs = []
        x = input
        for r, block in enumerate(self.blocks):
            if isinstance(block, ProgMultiBlock):
                y = self._forwardMulti(x, r, block)
            else:
                y = self._forwardSimple(x, r, block)
            outputs.append(y)
            x = y
        self.lastOutputList = outputs
        return outputs[-1]

    def _forwardSimple(self, x, row, block):
        currOutput = block.runBlock(x)
        if not block.isLateralized() or row == 0 or len(self.parentCols) < 1:
            y = block.runActivation(currOutput)
        else:
            for c, col in enumerate(self.parentCols):
                currOutput += block.runLateral(c, col.lastOutputList[row - 1])
            y = block.runActivation(currOutput)
        return y

    def _forwardMulti(self, x, row, block):
        currOutput = block.runBlock(x)
        if not block.isLateralized() or row == 0 or len(self.parentCols) < 1:
            y = block.runActivation(currOutput)
        else:
            for c, col in enumerate(self.parentCols):
                lats = block.runLateral(c, col.lastOutputList[row - 1])
                for i, p in enumerate(block.getPassDescriptor()):
                    if not p:   currOutput[i] += lats[i]
            y = block.runActivation(currOutput)
        return y
