import urllib

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.nn.parameter import Parameter
from torchvision import transforms

url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try:
    urllib.URLopener().retrieve(url, filename)
except:
    urllib.request.urlretrieve(url, filename)

input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def visualize_numpy(np_array: np.ndarray):
    """Visualize a numpy array.

    Args:
        np_array: The array to be visualized.
    """
    plt.imshow(np_array)
    plt.show()


def visualize_batch_input(batch_image: torch.Tensor):
    """Visualize the input batch image.

    Args:
        batch_image: The input batch image with shape (B, C, W, H)
    """
    batch_image = batch_image[0]
    numpy_input_tensor = np.transpose(batch_image.cpu().detach().numpy(), (1, 2, 0))
    img_mean = np.array([0.485, 0.456, 0.406])
    img_std = np.array([0.229, 0.224, 0.225])
    numpy_input_tensor = numpy_input_tensor * img_std + img_mean
    numpy_input_tensor = normalize_tensor(numpy_input_tensor)
    print(numpy_input_tensor.min(), numpy_input_tensor.max())
    visualize_numpy(numpy_input_tensor)


def tensor_numpy(input_tensor):
    batch_image = input_tensor[0]
    numpy_input_tensor = np.transpose(batch_image.cpu().detach().numpy(), (1, 2, 0))
    img_mean = np.array([0.485, 0.456, 0.406])
    img_std = np.array([0.229, 0.224, 0.225])
    numpy_input_tensor = numpy_input_tensor * img_std + img_mean

    numpy_input_tensor = normalize_tensor(numpy_input_tensor)

    return numpy_input_tensor


def normalize_tensor(input_tensor):
    input_tensor = (input_tensor - input_tensor.min()) / (input_tensor.max() - input_tensor.min())
    return input_tensor


def backward(forward_network: list, layer_index: int, input_batch: torch.Tensor):
    """Backward the forward network.

    Args:
        forward_network: The forward network.
        layer_index: The layer to be selected.
        input_batch: The input batch.
    """

    information_list = []
    output = input_batch

    for i in range(layer_index + 1):
        if i == layer_index:
            last = True
        else:
            last = False
        layer = forward_network[i]
        print(i, type(forward_network[i]))
        if type(layer) == torch.nn.modules.conv.Conv2d:
            output = layer(output)
            information_list.append(
                {
                    "label": "conv", "last": last,
                    "weight": layer.weight, "bias": layer.bias,
                    "info": {
                        "in_channels": layer.in_channels,
                        "out_channels": layer.out_channels,
                        "kernel_size": layer.kernel_size,
                        "padding": layer.padding,
                    }
                }
            )
        if type(layer) == torch.nn.modules.activation.ReLU:
            output = layer(output)
            information_list.append(
                {
                    "label": "relu", "last": last,
                    "weight": None, "bias": None,
                    "info": None,
                }
            )
        if type(layer) == torch.nn.modules.pooling.MaxPool2d:
            layer = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True).to(device)
            output, switches = layer(output)
            information_list.append(
                {
                    "label": "maxpool", "last": last,
                    "weight": None, "bias": None,
                    "info": switches,
                }
            )
        if type(layer) == torch.nn.modules.pooling.AdaptiveAvgPool2d:
            output = layer(output)
            information_list.append(
                {
                    "label": "adap", "last": last,
                    "weight": None, "bias": None,
                    "info": None,
                }
            )
        if type(layer) == torch.nn.modules.linear.Linear:
            output = layer(output)
            information_list.append(
                {
                    "label": "linear", "last": last,
                    "weight": layer.weight, "bias": layer.bias,
                    "info": {
                        "in_features": layer.in_features,
                        "out_features": layer.out_features,
                    }
                }
            )
        if type(layer) == torch.nn.modules.dropout.Dropout:
            information_list.append(
                {
                    "label": "drop", "last": last,
                    "weight": None, "bias": None,
                    "info": None,
                }
            )
        if type(layer) == str:
            information_list.append(
                {
                    "label": "flat", "last": last,
                    "weight": None, "bias": None,
                    "info": output.shape
                }
            )
            output = torch.flatten(output, 1)

    feature_dim = output.shape[1]

    print(torch.argmax(output))

    fig, ax = plt.subplots(nrows=8, ncols=8, figsize=(50, 50))

    counter = 0
    for row in ax:
        for col in row:
            index = [i for i in range(output.shape[1])]
            index.remove(counter)
            counter = counter + 1
            temp = output.clone()
            temp[:, index] = 0
            re_input = backward_one_channel(temp, information_list, layer_index)
            reinput_np = tensor_numpy(re_input)
            col.imshow(reinput_np)
    plt.show()


def backward_one_channel(zero_out_one, information_list, layer_index):
    for i in range(layer_index, -1, -1):
        information = information_list[i]
        if information["label"] == "conv":
            if not information["last"]:
                bias = information["bias"]
                bias = bias.view(1, -1, 1, 1)
                # zero_out_one = zero_out_one + bias
            deconv = construct_deconv(information)
            zero_out_one = deconv(zero_out_one)
        if information["label"] == "linear":
            delinear = construct_delinear(information)
            zero_out_one = delinear(zero_out_one)
        if information["label"] == "relu":
            derelu = construct_derelu()
            zero_out_one = derelu(zero_out_one)
        if information["label"] == "flat":
            zero_out_one = zero_out_one.reshape(information["info"])
        if information["label"] == "maxpool":
            demaxpool = construct_demaxpool()
            zero_out_one = demaxpool(zero_out_one, information["info"])
    return zero_out_one


def construct_demaxpool():
    demaxpool = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
    return demaxpool


def construct_derelu():
    derelu = nn.ReLU(True)
    return derelu


def construct_delinear(information):
    delinear = nn.Linear(
        in_features=information["info"]["out_features"],
        out_features=information["info"]["in_features"],
        bias=False,
    ).to(device)
    delinear.weight = Parameter(torch.transpose(information["weight"], 0, 1))
    return delinear


def construct_deconv(information):
    info = information["info"]
    in_channels = info["in_channels"]
    out_channels = info["out_channels"]
    kernel_size = info["kernel_size"]
    padding = info["padding"]
    de_conv_layer = nn.ConvTranspose2d(
        in_channels=out_channels, out_channels=in_channels,
        kernel_size=kernel_size, padding=padding
    ).to(device)
    de_conv_layer.weight = Parameter(torch.flip(information["weight"], [-2, -1]))
    return de_conv_layer


def main():
    """The main function to visualize the convolution filters.
    """
    layer_num = 4

    model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg11', pretrained=True)
    model.eval()
    model.to(device)

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    input_batch = input_batch.to(device)

    forward_network = []
    for feature_layer in model.features:
        forward_network.append(feature_layer)
    forward_network.append(model.avgpool)
    forward_network.append("flatten")
    for classifier_layer in model.classifier:
        forward_network.append(classifier_layer)

    backward(forward_network, layer_index=layer_num, input_batch=input_batch)


if __name__ == '__main__':
    main()



