import cv2
import onnx
from caffe2.python.onnx import backend
import onnxruntime as ort
from common.utils import drawLandmark_multiple
from vision.utils.feature_utils import eye_aspect_ratio
import torchvision.transforms as transforms
import onnxruntime
from detector.feature_detector import predict, get_box_mark, toimage
from svm.svm_predict import svm_predict
import argparse


if __name__ == '__main__':

    # rtmp = 'rtmp://dayang.link/live/test'
    rtmp = r'D:\DrowsinessData\YawDD dataset\Mirror\Male_mirror Avi Videos\40-MaleNoGlasses-Yawning.avi'

# environment:
    resize = transforms.Resize([112, 112])

    onnx_model_landmark = onnx.load("onnx/pfld.onnx")
    onnx.checker.check_model(onnx_model_landmark)
    ort_session_landmark = onnxruntime.InferenceSession("onnx/pfld.onnx")

    label_path = "models/voc-model-labels.txt"
    onnx_path = "models/onnx/version-RFB-320.onnx"
    class_names = [name.strip() for name in open(label_path).readlines()]

    predictor = onnx.load(onnx_path)
    onnx.checker.check_model(predictor)
    onnx.helper.printable_graph(predictor.graph)
    predictor = backend.prepare(predictor, device="CPU")

    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name

    cap = cv2.VideoCapture(rtmp)

# parameter:
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

    threshold_mouth = 0.40
    frame_mouth = 50
    counter_mouth = 0
    max_mouth = 0

    state_time = 0
    pred = 0

# process:
    while True:
        counter_time += 1

        ret, orig_image = cap.read()
        if orig_image is None:
            print("no img")
            break

        image = toimage(orig_image)
        confidences, boxes = ort_session.run(None, {input_name: image})
        boxes, labels, probs = predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, threshold)
        if boxes.shape[0] <= 0:
            print('No face detected')
            continue
        box = boxes[0]
        img = orig_image.copy()

        new_bbox, landmark = get_box_mark(img, box, ort_session, ort_session_landmark)

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

        # detect eyes' abnormal
        if EAR_left + EAR_right < threshold_eyes * 2:
            counter_eyes += 1
            flg_eye = 1
        else:
            flg_eye = 0

        # detect mouth's abnormal
        if EAR_mouth > threshold_mouth:
            if flg_mouth == 1:
                counter_mouth += 1
            else:
                flg_mouth = 1
            max_mouth = max(max_mouth, counter_mouth)
        else:
            flg_mouth = 0
            counter_mouth = 0

        orig_image = drawLandmark_multiple(orig_image, new_bbox, landmark)

        if counter_time >= threshold_time:
            # end of detecting and predict the result
            if flg_eye == 0 and flg_mouth == 0:
                counter_sum = counter_time + counter_extra
                P_70 = counter_eyes / counter_sum

                pred = svm_predict(P_70, max_mouth)[0]
                print('{:.2f} {}'.format(P_70, max_mouth) )
                print(pred)

                counter_time = 0
                counter_eyes = 0
                counter_extra = 0
                max_mouth = 0
            elif flg_eye != 0:
                counter_extra += 1

        if pred == 0:
            # normal state
            pass

        if pred == 1:
            # abnormal state
            pass

    cap.release()

