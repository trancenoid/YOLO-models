from torch import nn
from collections import OrderedDict
import torch
import numpy as np
from torchvision.ops import nms
from functools import lru_cache
from typing import Dict
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

class AnchorSet():
    def __init__(self, sizes):
        self.sizes = nn.Parameter(sizes, requires_grad=False)
    
    @classmethod
    def from_string(self, anchor_set_repr):
        return AnchorSet( torch.tensor( [tuple(int(y.strip()) for y in x[1:-1].split(' ') ) for x in anchor_set_repr.split("_")]) )

    def __repr__(self) -> str:
        return "_".join([ str(x.cpu().numpy()) for x in self.sizes])
    
class YOLOOutput(nn.Module):
    def __init__(self, anchors):
        super(YOLOOutput, self).__init__()
        self.anchors = anchors

    def forward(self, input):
        pass

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

    @torch.jit.ignore
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

    @torch.jit.ignore
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

    @torch.jit.ignore
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

    @torch.jit.ignore
    def iou(a, b, format = 'xywh'):
        '''
        return IOU between two boxes, each given as top-left coordinates,  (x1,y1,x2,y2)
        Deprecated in favor of torchvision inbuilt ops.nms ; No speed gain though :(
        '''
        assert format in ('xywh','xyxy')

        ax1,ay1,ax2,ay2 = a
        bx1,by1,bx2,by2 = b[:,0], b[:,1], b[:,2], b[:,3]

        device = a.device

        if format == 'xywh':
            ax2 = ax1+ax2
            ay2 = ay1+ay2
            bx2 = bx1+bx2
            by2 = by1+by2
        
        # union
        union_ = (ay2 - ay1)*(ax2 - ax1) + (by2 - by1)*(bx2 - bx1)
        union_ = torch.maximum(torch.tensor(1e-7, device=device), union_)

        # intersection
        intersection_x = torch.clamp(torch.minimum(ax2,bx2) - torch.maximum(ax1, bx1), min = torch.tensor(0, device=device), max = union_)
        intersection_y = torch.clamp(torch.minimum(ay2,by2) - torch.maximum(ay1, by1), min = torch.tensor(0, device=device), max = union_)
        intersection_ = intersection_y*intersection_x

        union_ = union_ - intersection_

        return intersection_/union_

    def _pred(output_dict : Dict[str, torch.Tensor], bbox_format : str = 'xywh', num_classes  : int= 80, OBJECT_CONFIDENCE : float = 0.7, 
                CLASS_CONFIDENCE : float = 0.5, NMS_THRESHOLD : float = 0.4):
        
        preds = torch.cat([x for x in output_dict.values()], dim = 1)
        preds = preds[0]
        assert len(preds.shape) == 2 , "preds shape must be (num_of_detections, num_classes+5)"

        # NMS
        detections = torch.empty((0,6), device = preds.device)
        confident_detections = torch.nonzero(torch.where(preds[:,4].gt(OBJECT_CONFIDENCE), 1.0, 0.0)).unbind(-1)
        confident_detections = preds[confident_detections[0], :]
        confident_classes = torch.nonzero(torch.where(confident_detections[:,-(num_classes):].gt(CLASS_CONFIDENCE), 1.0, 0.0)).unbind(-1)
        confident_detections = confident_detections[confident_classes[0]]
        for class_ in torch.unique(confident_classes[1], sorted=False):
            mask = confident_classes[1].eq(class_)
            curr_class_detections = confident_detections[mask,:]
            curr_class_detections[:,4] = curr_class_detections[:,5+class_]
            curr_class_detections[:,2:4] += curr_class_detections[:,:2]
            selected_boxes = nms(curr_class_detections[:,:4], curr_class_detections[:,4], NMS_THRESHOLD)
            class_rep = class_.repeat(selected_boxes.shape[0]).view(-1,1)
            _det = torch.cat( (curr_class_detections[selected_boxes,:5], class_rep ), dim=1 ) 
            detections = torch.cat( ( detections, _det ) )


        return detections
    

    def predict(self, x, **kwargs):
        '''
        x : batch of images (tensors) of shape ( bs, 3, H, W)
        Perform NMS and return (bs, N, 5) array where N is the no. of bounding boxes selected each having 4
        coordinates in supplied format and the class confidence score
        '''
        with torch.no_grad():
            x = self(x)

        return self._pred(x, **kwargs)
    


    @lru_cache
    def create_meshgrid(n_cellcols, n_cellrows, device):
        x = torch.arange(n_cellcols).view(1, -1)
        y = torch.arange(n_cellrows).view(-1, 1)

        # Broadcasting to create meshgrid
        cell_grid = torch.stack([x.repeat(n_cellrows, 1), y.repeat(1, n_cellcols)], dim=2).repeat(1,1,3).view(-1, 2).to(device=device)

        return cell_grid


    def logits_to_preds(x, num_classes, anchors : AnchorSet, img_dim):
        x = x.detach()
        device = x.device

        bbox_attrs = 4 + 1
        batch_size = x.shape[0]
        n_cellcols = x.shape[2]
        n_cellrows = x.shape[3]
        scale_factor = img_dim/n_cellcols

        cell_grid = YOLOv3.create_meshgrid(n_cellcols, n_cellrows, device)

        anchor_dims = anchors.sizes.repeat(n_cellrows*n_cellcols,1)

        # x : (Batch, BoxAttr x num_anchors , H , W)
        # reference : https://github.com/pytorch/vision/blob/main/torchvision/models/detection/ssd.py#L98
        x = x.view(batch_size, -1, bbox_attrs+num_classes, n_cellcols, n_cellrows)
        x = x.permute(0,3,4,1,2)
        x = x.reshape(batch_size, -1, bbox_attrs+num_classes)

        
        x[:,:,2:4] = torch.exp(x[:,:,2:4])*anchor_dims
        x[:,:,:2] = (torch.sigmoid(x[:,:,:2]) + cell_grid)*scale_factor - x[:,:,2:4]/2 
        x[:,:,4] = torch.sigmoid(x[:,:,4])
        x[:,:,bbox_attrs:] = torch.sigmoid(x[:,:,bbox_attrs:])

        return x
    
    def forward(self, x):
        cache = {}
        results = {}
        for i, block in enumerate(self.module_list):
            # string comparison is faster than isinstance
            blocktype = block[0].__class__.__name__
            if (blocktype in [nn.Conv2d.__name__, nn.Upsample.__name__]):
                x = block(x)
            elif blocktype == Shortcut.__name__:
                x = x + cache[block[0].route_from[0]]
            elif blocktype == Route.__name__:
                x = torch.cat(tuple(cache[t] for t in block[0].route_from), dim=1)
            elif blocktype == YOLOOutput.__name__:
                results[str(block[0].anchors)] = YOLOv3.logits_to_preds(x, num_classes = self.num_classes, 
                                   anchors = block[0].anchors, img_dim = self.img_dim)

            if i in self.layer_indices_to_store:
                cache[i] = x

        return results

    @torch.jit.ignore
    def create_model(self, device = 'cuda'):
        anchorset1 = AnchorSet(torch.tensor([(116, 90),  (156, 198),  (373, 326)], device=device))
        anchorset2 = AnchorSet(torch.tensor([(30, 61),  (62, 45),  (59, 119)], device=device))
        anchorset3 = AnchorSet(torch.tensor([(10, 13),  (16, 30),  (33, 23)], device= device))

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
                                                            YOLOOutput(anchors=anchorset1))])))

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
                                                            YOLOOutput(anchors=anchorset2))])))

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
                                                            YOLOOutput(anchors=anchorset3))])))

        assert len(self.module_list) == len(
            self.output_dims), f"At least one layer was not inserted properly, layers : {len(self.module_list)}, outputs dims seen : {len(self.output_dims)} "

    # taken from
    # https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/
    @torch.jit.ignore
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
