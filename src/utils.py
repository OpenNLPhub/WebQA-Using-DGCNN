import torch
from sklearn.metrics import confusion_matrix

'''--------------------- Loss Function ---------------------'''
def focal_loss(y_true,y_pred,device):
    alpha,gamma = torch.tensor(0.25).to(device) , torch.tensor(2.0).to(device)
    y_pred=torch.clamp(y_pred,1e-8,1-1e-8)
    return - alpha * y_true * torch.log(y_pred) * (1 - y_pred) ** gamma\
        - (1 - alpha) * (1 - y_true) * torch.log(1 -  y_pred) * y_pred



# def extract_answer(q_text,p_text_list)->dict:

def alignWord2Char(q_word):
    ans=[]
    for word in q_word:
        l=[ word for i in list(word)]
        ans.extend(l)
    return ans


# cal the confusion matrix

def binary_confusion_matrix_evaluate(y_true,y_pred):
    tn ,fp, fn, tp =  confusion_matrix(y_true,y_pred).ravel()
    acc = float(tn + tp)/(fp+fn+tn+tp) 
    prec =  float(tp) / (tp + fp) if (tp+fp) != 0 else 0.
    recall =  float(tp) / (tp + fn) if (tp + fn) != 0 else 0.
    f1= 2*prec*recall / ( prec + recall) if prec + recall !=0 else 0
    return acc,prec,recall,f1