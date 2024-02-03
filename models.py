from torch import nn
from collections import OrderedDict
import torch
import numpy as np
# refs
# https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/
# https://pjreddie.com/media/files/papers/YOLOv3.pdf

COCO_CLASSES = np.array(['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 
                'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 
                'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 
                'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
                'tennis racket','bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 
                'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 
                'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 
                'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 
                'teddy bear', 'hair drier', 'toothbrush'])


class YOLOOutput(nn.Module):
    def __init__(self, anchors):
        super(YOLOOutput, self).__init__()
        self.anchors = anchors


class Shortcut(nn.Module):
    def __init__(self, layer_index, layer_fanout):
        super(Shortcut, self).__init__()
        self.route_from = [layer_index]
        self.out_channels = layer_fanout

    def forward(self, input):
        pass


class Route(nn.Module):
    def __init__(self, layer_index_a, a_fanout, layer_index_b=None, b_fanout=None):
        super(Route, self).__init__()
        self.route_from = [layer_index_a]
        self.out_channels = a_fanout

        if layer_index_b is not None:
            self.route_from.append(layer_index_b)
            self.out_channels += b_fanout

    def forward(self, input):
        pass


class SpecialModuleList(nn.ModuleList):
    def __init__(self, output_dims, in_channels=3):
        super(SpecialModuleList, self).__init__()
        self.output_dims = output_dims
        self.last_dim = in_channels

    def append(self, element):
        '''
        Override super method to keep track of output channels for each layer added
        '''
        try:
            out_dim = element[0].out_channels
            self.last_dim = out_dim
        except AttributeError as e:
            out_dim = self.last_dim

        self.output_dims.append(out_dim)
        super(SpecialModuleList, self).append(element)

    def extend(self, elements):
        for element in elements:
            self.append(element)


class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes = 80, img_dim = 608):
        super(YOLOv3, self).__init__()
        # parameters 
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.img_dim = img_dim


        # Create DarkNet Body
        self.output_dims = []
        
        self.module_list = SpecialModuleList(self.output_dims, in_channels)

        self.create_model()

        # store layer index to cache (Shortcut and Route layers)
        self.layer_indices_to_store = []
        for block in self.module_list:
            if isinstance(block[0], (Shortcut, Route)):
                self.layer_indices_to_store.extend(block[0].route_from)


    def make_ConvStack(self, in_channels, out_channels, kernel_size, padding, stride, batchnorm, activation, name_suffix):
        module = nn.Sequential()
        if padding:
            padding = (kernel_size - 1) // 2
        else:
            padding = 0
        module.add_module(f"conv_{name_suffix}", nn.Conv2d(in_channels=in_channels,
                                                           out_channels=out_channels,
                                                           kernel_size=kernel_size,
                                                           padding=padding,
                                                           stride=stride,
                                                           bias=not batchnorm))
        if batchnorm:
            module.add_module(f"bn_{name_suffix}",
                              nn.BatchNorm2d(num_features=out_channels))

        if activation is not None and activation == 'leaky':
            module.add_module(f"leaky_{name_suffix}",
                              nn.LeakyReLU(negative_slope=0.1, inplace=True))

        return module

    def make_DarkNetBlock(self, in_channels, filters, sizes, batchnorm, name_suffix, skip_from: int):
        module_list = []

        module_list.append(self.make_ConvStack(in_channels=in_channels,
                                               out_channels=filters[0],
                                               kernel_size=sizes[0],
                                               padding=1,
                                               stride=1,
                                               batchnorm=batchnorm,
                                               activation='leaky',
                                               name_suffix=name_suffix))

        module_list.append(self.make_ConvStack(in_channels=filters[0],
                                               out_channels=filters[1],
                                               kernel_size=sizes[1],
                                               padding=1,
                                               stride=1,
                                               batchnorm=batchnorm,
                                               activation='leaky',
                                               name_suffix=name_suffix+1))
        module_list.append(nn.Sequential(OrderedDict(
            [(f"shortcut_{name_suffix+2}", Shortcut(skip_from, self.output_dims[skip_from]))])))

        return module_list

    def make_ConvBlock(self, in_channels, filters, sizes, batchnorm, name_suffix):
        module_list = []

        module_list.append(self.make_ConvStack(in_channels=in_channels,
                                               out_channels=filters[0],
                                               kernel_size=sizes[0],
                                               padding=1,
                                               stride=1,
                                               batchnorm=batchnorm,
                                               activation='leaky',
                                               name_suffix=name_suffix))

        module_list.append(self.make_ConvStack(in_channels=filters[0],
                                               out_channels=filters[1],
                                               kernel_size=sizes[1],
                                               padding=1,
                                               stride=1,
                                               batchnorm=batchnorm,
                                               activation='leaky',
                                               name_suffix=name_suffix+1))

        return module_list

    def pred_to_detections(preds): #WIP
        OBJECT_CONFIDENCE = 0.5
        CLASS_CONFIDENCE = 0.5

        anchors = [(116, 90), (156, 198), (373, 326)]
        # anchors = [(30, 61),  (62, 45),  (59, 119)]
        # anchors = [(10, 13),  (16, 30),  (33, 23)]

    def logits_to_preds(x, num_classes, anchors, img_dim):
        x = x.detach()
        device = x.device

        bbox_attrs = 4 + 1
        batch_size = x.shape[0]
        n_cellcols = x.shape[2]
        n_cellrows = x.shape[3]
        scale_factor = img_dim/n_cellcols

        cell_grid = torch.meshgrid(torch.arange(n_cellcols),torch.arange(n_cellrows), indexing='xy')
        cell_grid = torch.stack(cell_grid,dim=2).repeat(1,1,3).view(-1,2)

        anchor_dims = torch.Tensor(anchors, device = device).repeat(n_cellrows*n_cellcols,1)

        x = x.permute(0,2,3,1).contiguous()
        x = x.view(batch_size, n_cellcols*n_cellrows*len(anchors), (bbox_attrs + num_classes))

        x[:,:,:2] = (torch.sigmoid(x[:,:,:2]) + cell_grid)*scale_factor
        x[:,:,2:4] = torch.exp(x[:,:,2:4])*anchor_dims
        x[:,:,4] = torch.sigmoid(x[:,:,4])
        x[:,:,bbox_attrs:] = torch.sigmoid(x[:,:,bbox_attrs:])

        return x
    
    def forward(self, x):
        cache = {}
        results = {}
        for i, block in enumerate(self.module_list):
            if isinstance(block[0], (nn.Conv2d, nn.Upsample)):
                x = block(x)
            elif isinstance(block[0], Shortcut):
                x = x + cache[block[0].route_from[0]]
            elif isinstance(block[0], Route):
                x = torch.cat(tuple(cache[t]
                              for t in block[0].route_from), dim=1)
            elif isinstance(block[0], YOLOOutput):
                results[tuple(block[0].anchors)] = YOLOv3.logits_to_preds(x, num_classes = self.num_classes, 
                                   anchors = block[0].anchors, img_dim = self.img_dim)

            if i in self.layer_indices_to_store:
                cache[i] = x

        return results

    def create_model(self):
        # DarkNet-53 Body
        self.module_list.append(self.make_ConvStack(in_channels=self.in_channels,
                                                    out_channels=32,
                                                    kernel_size=3,
                                                    stride=1,
                                                    padding=1,
                                                    batchnorm=True,
                                                    activation='leaky',
                                                    name_suffix=0))

        specs = [1, 2, 8, 8, 4]
        in_channels = 32
        for i, reps in enumerate(specs):

            self.module_list.append(self.make_ConvStack(in_channels=in_channels,
                                                        out_channels=in_channels*2,
                                                        kernel_size=3,
                                                        stride=2,
                                                        padding=1,
                                                        batchnorm=True,
                                                        activation='leaky',
                                                        name_suffix=len(self.module_list)))

            for _ in range(reps):
                self.module_list.extend(self.make_DarkNetBlock(in_channels=in_channels*2,
                                                               filters=[
                                                                   in_channels, in_channels*2],
                                                               sizes=[1, 3],
                                                               batchnorm=True,
                                                               name_suffix=len(
                                                                   self.module_list),
                                                               skip_from=len(self.module_list)-1))

            in_channels = in_channels*2

        # Create YOLO heads
        #### HEAD 1 ####
        for _ in range(3):
            self.module_list.extend(self.make_ConvBlock(in_channels=self.module_list[-1][0].out_channels,
                                                        filters=[512, 1024],
                                                        sizes=[1, 3],
                                                        batchnorm=True,
                                                        name_suffix=len(self.module_list)))

        self.module_list.append(self.make_ConvStack(in_channels=self.module_list[-1][0].out_channels,
                                                    out_channels=255,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=1,
                                                    batchnorm=False,
                                                    activation=None,
                                                    name_suffix=len(self.module_list)))

        self.module_list.append(nn.Sequential(OrderedDict([(f"yolo_{len(self.module_list)}",
                                                            YOLOOutput(anchors=[(116, 90),  (156, 198),  (373, 326)]))])))

        #### HEAD 2 ####
        self.module_list.append(nn.Sequential(OrderedDict(
            [(f"route_{len(self.module_list)}", Route(len(self.module_list)-4, self.output_dims[-4]))])))

        self.module_list.append(self.make_ConvStack(in_channels=self.module_list[-1][0].out_channels,
                                                    out_channels=256,
                                                    kernel_size=1,
                                                    padding=1,
                                                    stride=1,
                                                    batchnorm=True,
                                                    activation='leaky',
                                                    name_suffix=len(self.module_list)))

        self.module_list.append(nn.Sequential(OrderedDict(
            [(f"upsample_{len(self.module_list)}", nn.Upsample(scale_factor=2, mode='nearest'))])))

        self.module_list.append(nn.Sequential(OrderedDict([(f"route_{len(self.module_list)}", Route(len(self.module_list)-1,
                                                                                                    self.output_dims[-1],
                                                                                                    61, self.output_dims[61]))])))
        for _ in range(3):
            self.module_list.extend(self.make_ConvBlock(in_channels=self.module_list[-1][0].out_channels,
                                                        filters=[256, 512],
                                                        sizes=[1, 3],
                                                        batchnorm=True,
                                                        name_suffix=len(self.module_list)))

        self.module_list.append(self.make_ConvStack(in_channels=self.module_list[-1][0].out_channels,
                                                    out_channels=255,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=1,
                                                    batchnorm=False,
                                                    activation=None,
                                                    name_suffix=len(self.module_list)))

        self.module_list.append(nn.Sequential(OrderedDict([(f"yolo_{len(self.module_list)}",
                                                            YOLOOutput(anchors=[(30, 61),  (62, 45),  (59, 119)]))])))

        #### HEAD 3 ####
        self.module_list.append(nn.Sequential(OrderedDict(
            [(f"route_{len(self.module_list)}", Route(len(self.module_list)-4, self.output_dims[-4]))])))

        self.module_list.append(self.make_ConvStack(in_channels=self.module_list[-1][0].out_channels,
                                                    out_channels=128,
                                                    kernel_size=1,
                                                    padding=1,
                                                    stride=1,
                                                    batchnorm=True,
                                                    activation='leaky',
                                                    name_suffix=len(self.module_list)))

        self.module_list.append(nn.Sequential(OrderedDict(
            [(f"upsample_{len(self.module_list)}", nn.Upsample(scale_factor=2, mode='nearest'))])))

        self.module_list.append(nn.Sequential(OrderedDict([(f"route_{len(self.module_list)}", Route(len(self.module_list)-1,
                                                                                                    self.output_dims[-1],
                                                                                                    36, self.output_dims[36]))])))
        for _ in range(3):
            self.module_list.extend(self.make_ConvBlock(in_channels=self.module_list[-1][0].out_channels,
                                                        filters=[128, 256],
                                                        sizes=[1, 3],
                                                        batchnorm=True,
                                                        name_suffix=len(self.module_list)))

        self.module_list.append(self.make_ConvStack(in_channels=self.module_list[-1][0].out_channels,
                                                    out_channels=255,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=1,
                                                    batchnorm=False,
                                                    activation=None,
                                                    name_suffix=len(self.module_list)))

        self.module_list.append(nn.Sequential(OrderedDict([(f"yolo_{len(self.module_list)}",
                                                            YOLOOutput(anchors=[(10, 13),  (16, 30),  (33, 23)]))])))

        assert len(self.module_list) == len(
            self.output_dims), f"At least one layer was not inserted properly, layers : {len(self.module_list)}, outputs dims seen : {len(self.output_dims)} "

    # taken from
    # https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/
    def load_weights(self, weightfile):
        # Open the weights file
        fp = open(weightfile, "rb")

        # The first 5 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(fp, dtype=np.float32)

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = "convolutional" if isinstance(
                self.module_list[i][0], nn.Conv2d) else "Other"

            # If module_type is convolutional load weights
            # Otherwise ignore.

            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = isinstance(
                        self.module_list[i][1], nn.BatchNorm2d)
                except:
                    batch_normalize = 0

                conv = model[0]

                if (batch_normalize):
                    bn = model[1]

                    # Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

                    # Load the weights
                    bn_biases = torch.from_numpy(
                        weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(
                        weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(
                        weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(
                        weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # Cast the loaded weights into dims of model weights.
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    # Number of biases
                    num_biases = conv.bias.numel()

                    # Load the weights
                    conv_biases = torch.from_numpy(
                        weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    # reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # Finally copy the data
                    conv.bias.data.copy_(conv_biases)

                # Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()

                # Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)
