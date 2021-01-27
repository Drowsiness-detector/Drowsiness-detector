from detector.feature_detector import feature_detector
from svm.svm_predict import svm_predict

path = r'D:\DrowsinessData\YawDD dataset\Mirror\Male_mirror Avi Videos\40-MaleNoGlasses-Yawning.avi'
eyes, mouth = feature_detector(path)
pred = svm_predict(eyes, mouth)
print(pred)
