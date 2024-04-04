import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def compute_iou(box1, box2):
    """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    """
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(
        box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)
    return iou


class YoloLoss(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def xywh2xyxy(self, boxes):
        """
        Parameters:
        boxes: (N,4) representing by x,y,w,h

        Returns:
        boxes: (N,4) representing by x1,y1,x2,y2

        if for a Box b the coordinates are represented by [x, y, w, h] then
        x1, y1 = x/S - 0.5*w, y/S - 0.5*h ; x2,y2 = x/S + 0.5*w, y/S + 0.5*h
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        """
        ### CODE ###
        # Your code here
        x1 = boxes[:, 0] / self.S - boxes[:, 2] / 2
        y1 = boxes[:, 1] / self.S - boxes[:, 3] / 2
        x2 = boxes[:, 0] / self.S + boxes[:, 2] / 2
        y2 = boxes[:, 1] / self.S + boxes[:, 3] / 2
        
        boxes = torch.stack((x1, y1, x2, y2), dim=1)

        return boxes

    def find_best_iou_boxes(self, pred_box_list, box_target):
        """
        Parameters:
        box_pred_list : (list) [(tensor) size (-1, 5)]  
        box_target : (tensor)  size (-1, 4)

        Returns:
        best_iou: (tensor) size (-1, 1)
        best_boxes : (tensor) size (-1, 5), containing the boxes which give the best iou among the two (self.B) predictions

        Hints:
        1) Find the iou's of each of the 2 bounding boxes of each grid cell of each image.
        2) For finding iou's use the compute_iou function
        3) use xywh2xyxy to convert bbox format if necessary,
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        We perform this transformation to convert the correct coordinates into bounding box coordinates.
        """
        N = box_target.size(0)
        box_target = self.xywh2xyxy(box_target)

        best_ious = torch.zeros(N).to('cuda')
        best_boxes = torch.zeros(N, 5).to('cuda')

        ious = torch.zeros(N, self.B).to('cuda')

        for b in range(len(pred_box_list)):
            iou = compute_iou(self.xywh2xyxy(pred_box_list[b][:, :4]), box_target) # (N, N)
            # get diagonal of iou matrix iou[i][j]
            iou = iou.diag() # (N, )
            ious[:, b] = iou

        for n in range(N):
            best_ious[n] = torch.max(ious[n])
            best_boxes[n] = pred_box_list[torch.argmax(ious[n])][n]

        # detach best_ious so it will not be involved in backward gradient calculation
        best_ious = best_ious.unsqueeze(1).detach()

        return best_ious, best_boxes



    def get_class_prediction_loss(self, classes_pred, classes_target, has_object_map):
        """
        Parameters:
        classes_pred : (tensor) size (batch_size, S, S, 20)
        classes_target : (tensor) size (batch_size, S, S, 20)
        has_object_map: (tensor) size (batch_size, S, S)

        Returns:
        class_loss : scalar
        """
        ### CODE ###
        # Your code here
        classes_sum = has_object_map * (classes_pred - classes_target).pow(2).sum(dim=-1) # (batch_size, S, S)
        loss = classes_sum.sum()

        return loss

    def get_no_object_loss(self, pred_boxes_list, has_object_map):
        """
        Parameters:
        pred_boxes_list: (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        has_object_map: (tensor) size (N, S, S)

        Returns:
        loss : scalar

        Hints:
        1) Only compute loss for cell which doesn't contain object
        2) compute loss for all predictions in the pred_boxes_list list
        3) You can assume the ground truth confidence of non-object cells is 0
        """
        ### CODE ###
        # Your code here
        no_obj_loss = 0
        has_no_obj_map = has_object_map == 0

        for b in range(len(pred_boxes_list)):
            conf = pred_boxes_list[b][:, :, :, 4][has_no_obj_map] # (N, S, S)
            no_obj_loss += F.mse_loss(conf, torch.zeros_like(conf), reduction='sum')

        no_obj_loss *= self.l_noobj
        return no_obj_loss

        # return loss

    def get_contain_conf_loss(self, box_pred_conf, box_target_conf):
        """
        Parameters:
        box_pred_conf : (tensor) size (-1,1)
        box_target_conf: (tensor) size (-1,1)

        Returns:
        contain_loss : scalar

        Hints:
        The box_target_conf should be treated as ground truth, i.e., no gradient

        """
        ### CODE
        # your code here
        return F.mse_loss(box_pred_conf, box_target_conf, reduction='sum')

    def get_regression_loss(self, box_pred_response, box_target_response):
        """
        Parameters:
        box_pred_response : (tensor) size (-1, 4)
        box_target_response : (tensor) size (-1, 4)
        Note : -1 corresponds to ravels the tensor into the dimension specified
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as

        Returns:
        reg_loss : scalar

        """
        ### CODE
        # your code here
        box_pred_response[:, 2:] = box_pred_response[:, 2:].sqrt()
        box_target_response[:, 2:] = box_target_response[:, 2:].sqrt()

        reg_loss =  self.l_coord * F.mse_loss(box_pred_response, box_target_response, reduction='sum')
        return reg_loss

    def forward(self, pred_tensor, target_boxes, target_cls, has_object_map):
        """
        pred_tensor: (tensor) size(N,S,S,Bx5+20=30) where:  
                            N - batch_size
                            S - width/height of network output grid
                            B - number of bounding boxes this grid cell is a part of = 2
                            5 - number of bounding box values corresponding to [x, y, w, h, c]
                                where x - x_coord, y - y_coord, w - width, h - height, c - confidence of having an object
                            20 - number of classes

        target_boxes: (tensor) size (N, S, S, 4): the ground truth bounding boxes
        target_cls: (tensor) size (N, S, S, 20): the ground truth class
        has_object_map: (tensor, bool) size (N, S, S): the ground truth for whether each cell contains an object (True/False)

        Returns:
        loss_dict (dict): with key value stored for total_loss, reg_loss, containing_obj_loss, no_obj_loss and cls_loss
        """
        N = pred_tensor.size(0)
        total_loss = 0.0

        # split the pred tensor from an entity to separate tensors:
        # -- pred_boxes_list: a list containing all bbox prediction (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        # -- pred_cls (containing all classification prediction)
        # your code here
        pred_boxes_list = []
        pred_cls = pred_tensor[:, :, :, self.B * 5:] # (N, S, S, 20)

        for b in range(self.B):
            start, end = b * 5, (b + 1) * 5
            pred_boxes_list.append(pred_tensor[:, :, :, start : end])

        # compcute classification loss
        # your code here
        cls_loss = self.get_class_prediction_loss(pred_cls, target_cls, has_object_map)

        # compute no-object loss
        # your code here
        no_obj_loss = self.get_no_object_loss(pred_boxes_list, has_object_map)

        # Re-shape boxes in pred_boxes_list and target_boxes to meet the following desires
        # 1) only keep having-object cells
        # 2) vectorize all dimensions except for the last one for faster computation
        # your code here
        for i in range(len(pred_boxes_list)):
            # reshape the boxes to meet the desire, and keep only the cells which have objects
            pred_boxes_list[i] = pred_boxes_list[i][has_object_map] # (-1, 5)
        
        has_obj_target_boxes = target_boxes[has_object_map] # (-1, 4)

        # find the best boxes among the 2 (or self.B) predicted boxes and the corresponding iou
        # your code here
        best_ious, best_boxes = self.find_best_iou_boxes(pred_boxes_list, has_obj_target_boxes)

        # compute regression loss between the found best bbox and GT bbox for all the cell containing objects
        # your code here
        reg_loss = self.get_regression_loss(best_boxes[:, :4], has_obj_target_boxes)

        # compute contain_object_loss
        # your code here
        contain_obj_loss = self.get_contain_conf_loss(best_boxes[:, 4].unsqueeze(-1), best_ious)

        # compute final loss
        # your code here
        total_loss += (cls_loss + no_obj_loss + contain_obj_loss + reg_loss) / N

        # construct return loss_dict
        # your code for loss_dict here
        loss_dict = dict(
            total_loss=total_loss,
            reg_loss=reg_loss,
            containing_obj_loss=contain_obj_loss,
            no_obj_loss=no_obj_loss,
            cls_loss=cls_loss,
        )
        return loss_dict
