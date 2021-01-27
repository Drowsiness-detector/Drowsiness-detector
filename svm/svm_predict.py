import pickle

def svm_predict(eyes ,mouth):
    with open(r'C:\Users\14595\Desktop\docker\svm\svm_model_7.pickle','rb') as f:
        model = pickle.load(f)
        pred = model.predict([[eyes, mouth]])
        return pred




