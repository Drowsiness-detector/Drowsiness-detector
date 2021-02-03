import vision.utils.box_utils_numpy as box_utils
import cv2
import numpy as np
from common.utils import BBox
from PIL import Image
import torchvision.transforms as transforms


def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):

    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = box_utils.hard_nms(box_probs,
                                       iou_threshold=iou_threshold,
                                       top_k=top_k,
                                       )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


def to_numpy(tensor):

    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def get_box_mark(img, box, ort_session, ort_session_landmark):

    out_size = 56
    resize = transforms.Resize([112, 112])
    to_tensor = transforms.ToTensor()

    height, width, _ = img.shape
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    size = int(max([w, h]) * 1.1)
    cx = x1 + w // 2
    cy = y1 + h // 2
    x1 = cx - size // 2
    x2 = x1 + size
    y1 = cy - size // 2
    y2 = y1 + size
    dx = max(0, -x1)
    dy = max(0, -y1)
    x1 = max(0, x1)
    y1 = max(0, y1)

    edx = max(0, x2 - width)
    edy = max(0, y2 - height)
    x2 = min(width, x2)
    y2 = min(height, y2)
    new_bbox = list(map(int, [x1, x2, y1, y2]))
    new_bbox = BBox(new_bbox)
    cropped = img[new_bbox.top:new_bbox.bottom, new_bbox.left:new_bbox.right]
    if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
        cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)
    cropped_face = cv2.resize(cropped, (out_size, out_size))

    if cropped_face.shape[0] <= 0 or cropped_face.shape[1] <= 0:
        return
    cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
    cropped_face = Image.fromarray(cropped_face)
    test_face = resize(cropped_face)
    test_face = to_tensor(test_face)
    test_face.unsqueeze_(0)

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(test_face)}
    ort_outs = ort_session_landmark.run(None, ort_inputs)
    landmark = ort_outs[0]
    landmark = landmark.reshape(-1, 2)
    landmark = new_bbox.reprojectLandmark(landmark)

    return new_bbox, landmark


def toimage(orig_image):

    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (320, 240))
    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)

    return image