import torch
import numpy as np

from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn


# def load_inference_resnet50(gpu_device_name=None):
#     # instantiate model either on cpu or gpu
#     device_name = "cpu" if gpu_device_name is None else gpu_device_name
#     device = torch.device(device_name)

#     # load pretrained model
#     model = fasterrcnn_resnet50_fpn(pretrained=True)
#     #model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained = True)
#     # set model to inference mode
#     for param in model.parameters():
#         param.requires_grad = False
#     with torch.no_grad():
#         model.eval()

#     return model.to(device)


# def preprocess_input(image):
#     transformer = transforms.ToTensor()
#     tensor_image = transformer(image)
#     batch_input = tensor_image.unsqueeze(0)
#     return batch_input


# def filter_objects(model_output, coco_cat_names, threshold):
#     bboxes = model_output["boxes"].numpy().astype("int")
#     labels = model_output["labels"].numpy()
#     scores = model_output["scores"].numpy()

#     idxs = np.where(np.isin(labels, coco_cat_names) & (scores >= threshold))
#     filtered_bboxes = bboxes[idxs]
#     return filtered_bboxes


# def detect_objects(model, image, coco_cat_names, threshold=0):
#     image_input = preprocess_input(image)
#     output = model(image_input)[0]
#     #  returning only the bounding boxes of the objects (car) that meet the specified criteria.
#     bboxes = filter_objects(output, coco_cat_names, threshold)
#     return bboxes

def detect_cars(model, image, coco_cat_names, threshold = 0):
    # Run inference 
    results = model(image)

    # Get bounding box 
    boxes, confidences, class_ids = results.pred[0][:, :4], results.pred[0][:, 4], results.pred[0][:, 5]

    # Filter out the class_id = [3, 4]
    relevant_indices = np.where(np.isin(class_ids, coco_cat_names))

    filtered_boxes = boxes[relevant_indices]

    for box in filtered_boxes:
        box = list(map(int, box))

    return filtered_boxes




def fetch_centroids(bounding_boxes):
    centroids = np.c_[
        bounding_boxes[:, [0, 2]].mean(axis=1),
        bounding_boxes[:, [1, 3]].mean(axis=1)]
    return centroids.astype("int")


def intersection_over_union(poly_1, poly_2):
    inter_area = poly_1.intersection(poly_2).area
    iou = inter_area / (poly_1.area + poly_2.area - inter_area)
    return iou


def is_occupied(spot, candidate, threshold=0.45):
    iou = intersection_over_union(spot, candidate)
    return iou > threshold
