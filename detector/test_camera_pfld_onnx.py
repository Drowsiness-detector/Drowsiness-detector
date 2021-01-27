import time
import cv2
import numpy as np
import onnx
import vision.utils.box_utils_numpy as box_utils
from caffe2.python.onnx import backend
import onnxruntime as ort
from common.utils import BBox,drawLandmark,drawLandmark_multiple
from vision.utils.feature_utils import eye_aspect_ratio
from PIL import Image
import torchvision.transforms as transforms
import onnxruntime
from svm.svm_predict import svm_predict

resize = transforms.Resize([112, 112])
to_tensor = transforms.ToTensor()

# 特征点模型
onnx_model_landmark = onnx.load("../onnx/pfld.onnx")
onnx.checker.check_model(onnx_model_landmark)
ort_session_landmark = onnxruntime.InferenceSession("../onnx/pfld.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# 特征点检测
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
    # （boxes, labels, probs）

label_path = "../models/voc-model-labels.txt"
onnx_path = "../models/onnx/version-RFB-320.onnx"
class_names = [name.strip() for name in open(label_path).readlines()]

predictor = onnx.load(onnx_path)
onnx.checker.check_model(predictor)     # 检查模型大小
onnx.helper.printable_graph(predictor.graph)
predictor = backend.prepare(predictor, device="CPU")  # default CPU

ort_session = ort.InferenceSession(onnx_path)   # 加载人脸模型
input_name = ort_session.get_inputs()[0].name

# cap = cv2.VideoCapture(0)  # capture from camera
# cap = cv2.VideoCapture(r'D:\DrowsinessData\YawDD dataset\Mirror\Male_mirror Avi Videos\1-MaleNoGlasses-Talking.avi')
# cap = cv2.VideoCapture(r'D:\DrowsinessData\YawDD dataset\Mirror\Female_mirror\1-FemaleNoGlasses-Talking.avi')
# cap = cv2.VideoCapture(r'D:\DrowsinessData\YawDD dataset\Mirror\Female_mirror\1-FemaleNoGlasses-Yawning.avi')
# cap = cv2.VideoCapture(r'D:\DrowsinessData\YawDD dataset\Mirror\Female_mirror\10-FemaleNoGlasses-Yawning.avi')
cap = cv2.VideoCapture(r'D:\DrowsinessData\YawDD dataset\Mirror\Male_mirror Avi Videos\40-MaleNoGlasses-Yawning.avi')
# cap = cv2.VideoCapture(r'D:\DrowsinessData\YawDD dataset\Mirror\Male_mirror Avi Videos\3-MaleNoGlasses-Yawning.avi')
# cap = cv2.VideoCapture(r'D:\DrowsinessData\YawDD dataset\Mirror\Female_mirror\25-FemaleSunGlasses-Yawning.avi')

threshold = 0.7
sum = 0

counter_time = 0
threshold_time = 60
counter_extra = 0
flg_eye = 0
flg_mouth = 0

threshold_eyes = 0.25
frame_eyes = 48
counter_eyes = 0
# 假设每两秒钟传一次， 30帧一秒， P80 = 48

threshold_mouth = 0.40
frame_mouth = 50
counter_mouth = 0
max_mouth = 0

# state = ''
state_time = 0
pred = 0
# fix = 5

while True:
    counter_time += 1

    ret, orig_image = cap.read()
    if orig_image is None:
        print("no img")
        break

    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (320, 240))
    # image = cv2.resize(image, (640, 480))
    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    # 处理图像

    time_time = time.time()
    confidences, boxes = ort_session.run(None, {input_name: image})
    # 预测 人脸模型
    # print("face_detect cost time:{}".format(time.time() - time_time))
    boxes, labels, probs = predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, threshold)
    # 预测 特征点 （width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1）

    # print(boxes.shape[0])
    # 对每个人脸头像
    # for i in range(boxes.shape[0]):
    i = 0
    box = boxes[i, :]
    label = f"{class_names[labels[i]]}: {probs[i]:.2f}"

    #cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
    # perform landmark detection
    out_size = 56
    img=orig_image.copy()
    height,width,_=img.shape
    x1=box[0]
    y1=box[1]
    x2=box[2]
    y2=box[3]
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    size = int(max([w, h])*1.1)
    cx = x1 + w//2
    cy = y1 + h//2
    x1 = cx - size//2
    x2 = x1 + size
    y1 = cy - size//2
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
    cropped=img[new_bbox.top:new_bbox.bottom,new_bbox.left:new_bbox.right]
    if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
        cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)
    cropped_face = cv2.resize(cropped, (out_size, out_size))

    if cropped_face.shape[0]<=0 or cropped_face.shape[1]<=0:
        continue
    cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
    cropped_face = Image.fromarray(cropped_face)
    test_face = resize(cropped_face)
    test_face = to_tensor(test_face)
    #test_face = normalize(test_face)
    test_face.unsqueeze_(0)

    start = time.time()
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(test_face)}
    ort_outs = ort_session_landmark.run(None, ort_inputs)
    end = time.time()
    # print('landmark cost Time: {:.6f}s.'.format(end - start))
    landmark = ort_outs[0]
    landmark = landmark.reshape(-1,2)
    landmark = new_bbox.reprojectLandmark(landmark)

    # landmark = landmark[36:48] # eyes
    # landmark = landmark[27:36] # nose
    # landmark = landmark[48:60] # mouth outside
    # landmark = landmark[60:68] # mouth inside
    # print(len(landmark))

    eyes_left = []
    eyes_right = []
    mouth = []
    for n in range(36, 42):
        eyes_left.append(landmark[n])
    for n in range(42, 48):
        eyes_right.append(landmark[n])
    for n in [60, 61, 63, 64, 65, 67]:
        mouth.append(landmark[n])
    EAR_left = eye_aspect_ratio(eyes_left)
    EAR_right = eye_aspect_ratio(eyes_right)
    EAR_mouth = eye_aspect_ratio(mouth)


    if EAR_left+EAR_right < threshold_eyes * 2:
        counter_eyes += 1
        flg_eye = 1
        cv2.putText(orig_image, "Eyes:abnormal", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0 ), 2)
        # if counter_eyes >= frame_eyes:
        #     cv2.putText(orig_image, "****************ALERT!****************", (10, 30),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 2)
        #     cv2.putText(orig_image, "****************ALERT!****************", (10, 325),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 2)
    else:
        flg_eye = 0

    if EAR_mouth > threshold_mouth:
        cv2.putText(orig_image, "mouth:abnormal", (10, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if flg_mouth == 1:
            counter_mouth += 1
        else:
            flg_mouth = 1
        max_mouth = max(max_mouth, counter_mouth)
        # if counter_mouth >= frame_mouth:
        #     cv2.putText(orig_image, "****************ALERT!****************", (10, 30),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 2)
        #     cv2.putText(orig_image, "****************ALERT!****************", (10, 325),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 2)
    else:
        flg_mouth = 0
        counter_mouth = 0

    orig_image = drawLandmark_multiple(orig_image, new_bbox, landmark)
    cv2.putText(orig_image, 'EAR_left:{:.2f}'.format(EAR_left), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(orig_image, 'EAR_right:{:.2f}'.format(EAR_right), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(orig_image, 'EAR_mouth:{:.2f}'.format(EAR_mouth), (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    if counter_time >= threshold_time:
        if flg_eye == 0 and flg_mouth == 0:
            counter_sum = counter_time + counter_extra
            P_70 = counter_eyes / counter_sum
            # if P_80 >= 0.8:
                # eyes indeed
            # if max_mouth >= frame_mouth:
                # mouth indeed
            pred = svm_predict(P_70, max_mouth)[0]
            # print('{:.2f} {}'.format(P_70, max_mouth) )
            # print(pred)
            counter_time = 0
            counter_eyes = 0
            counter_extra = 0
            max_mouth = 0
            # pred
        elif flg_eye != 0 :
            counter_extra += 1

    if pred == 0:
        cv2.putText(orig_image, 'State: Normal', (200, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0),
                    2)
    if pred == 1:
        cv2.putText(orig_image, 'State: Drowsiness', (200, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255),
                    2)


    sum += boxes.shape[0]
    orig_image = cv2.resize(orig_image, (0, 0), fx=1.0, fy=1.0)
    cv2.imshow('annotated', orig_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
# print("sum_frames:{}".format(sum))
