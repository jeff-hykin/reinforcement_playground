from agents.informed_vae.main import ImageEncoder

import torch
from tools.file_system_tools import FS
from tools.dataset_tools import Mnist
from tools.pytorch_tools import read_image, to_tensor
from torchvision import transforms
import torch.nn.functional as F

# preprocess data, and make it an iterator of input-output pairs
transformed_mnist_pairs = Mnist(
    # convert image-tensor to a float tensor then normalize
    transform_input=lambda each_image: transforms.Normalize((0.5,), (0.5,))(each_image.type(torch.float)),
    # convert number to one-hot encoding
    transform_output=lambda each_number: F.one_hot(to_tensor(each_number), num_classes=Mnist.number_of_classes).type(torch.float),
)

dummy_encoder = ImageEncoder()
encoded_output = dummy_encoder.forward(transformed_mnist_pairs[0][0])
print((encoded_output * 1000).type(torch.int))
# TODO: why would this value not be close to uniform?

print('training on Mnist')
dummy_encoder.fit(transformed_mnist_pairs)
for each in [ 31, 40 ]:
    print('actual: ', transformed_mnist_pairs[each][1])
    print('predicted: ', (dummy_encoder.forward(transformed_mnist_pairs[each][0]) * 1000).type(torch.int))
