from inspect import indentsize
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from .forms import UploadFileForm
import pickle
import numpy as np
import pandas as pd

import shap
from shap import Explanation
import math

import matplotlib.pyplot as plt


def app(request):
    # print("referrer",request.META.get('HTTP_REFERER'))
    # print(tf.config.list_physical_devices('GPU'))
    # physical_devices = tf.config.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # for plot in glob.glob('static/data/plots/*/*'):
    #     os.remove(plot)

    # paths = glob.glob('static/data/samples/*')
    # filenames = [i.split('/')[-1] for i in paths]
    

    return render(request,'index.html')

   

def index(request):
    OTP = ""
    print('referer',request.META.get('HTTP_REFERER'))
    print('method',request.method )
    if request.method == 'POST' and request.META.get('HTTP_REFERER') :

        email = request.POST.get('email')
        #print(email, request.POST)
        with open('static/data/users.txt','r') as fr:
            d = fr.read()
            if(email not in d):
                with open('static/data/users.txt','a+') as f:
                    f.write(email+'\n')

        port = 465  # For SSL
        smtp_server = "smtp.gmail.com"
        sender_email = "ecg.sbilab@gmail.com" 
        with open('static/data/config.dat','r') as f:
            password = f.read()

        digits = "0123456789"
        
        for i in range(4) :
            OTP += digits[math.floor(np.random.random() * 10)]

        message = """\
        Subject: OTP For ECG Analyzer Login

        Your OTP for login is : """+str(OTP)
        
        
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, email, message)

    
        return render(request, 'home.html', {'otp':OTP})
    else:
        return render(request,'home.html')


@csrf_exempt
def handle_uploaded_file(f):
    with open('static/data/uploadedtest.csv', 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)

@csrf_exempt
def upload(request):


    if request.method == 'POST':

        form = UploadFileForm(request.POST, request.FILES)
        
        if form.is_valid():

            handle_uploaded_file(request.FILES['file'])

        else:
            form = UploadFileForm()
            

    return render(request, 'index.html')

def classifiy(inputs, clf, clf2_1, clf2_2):
    
    labels = clf.predict(inputs)
    labels1 = clf2_1.predict(inputs)
    labels2 = clf2_2.predict(inputs)
    
    fl = np.zeros(labels.shape)
    fl[np.where(labels == 0)] = labels1[np.where(labels == 0)]
    fl[np.where(labels == 1)] = (labels2[np.where(labels == 1)] + 2)
    
    return fl

def getpredprobs(inputs, clf, clf2_1, clf2_2):
    
    labels = clf.predict(inputs)
    probs = clf.predict_proba(inputs)
    probs1 = clf2_1.predict_proba(inputs) * probs[:,0].reshape(-1,1)
    probs2 = clf2_2.predict_proba(inputs) * probs[:,1].reshape(-1,1)
    
    return np.concatenate((probs1,probs2), axis=1)

def predict(request):

    #return render(request, 'result.html' ) 
 
    feats = ['Age', 'duration of stay', 'Hemoglobin', 'Platelet Count', 'MCV',
       'MCHC', 'RDW', 'Monocytes', 'Neutrophils Abs', 'Lymphocytes Abs',
       'Eosinophils- Abs', 'Monocytes-Abs', 'Basophils-Abs', 'NLR', 'LMR',
       'NMR', 'TOTAL BILIRUBIN', 'INDIRECT BILIRUBIN.', 'SGPT/ALT', 'SGOT/AST',
       'TOTAL PROTEIN', 'ALKALI PHOSPHATASE', 'GLOBULIN', 'A/G Ratio',
       'Albumin', 'UREA', 'CREATINI', 'CALCIUM', 'PHOSPHOROUS', 'SODIUM',
       'POTASSIUM', 'CHLORIDE(CL-)', 'Uric Acid', 'Ferritin', 'LDH', 'IL-6',
       'CRp-alb', 'D - Dimer', 'Fibrinogen', 'INR', 'HbA1c']
    
    if (request.method == 'POST'):
        data = list(request.POST.values())
        data.pop(0)
        values  = np.array(data).reshape(1,41)
        print(values)
    
    with open('static/data/modelA.pkl', 'rb') as f:
        clf = pickle.load(f)
    with open('static/data/modelB.pkl', 'rb') as f:
        clf2_1 = pickle.load(f)
    with open('static/data/modelC.pkl', 'rb') as f:
        clf2_2 = pickle.load(f)
        
    out = classifiy(values, clf, clf2_1, clf2_2)
    probs = getpredprobs(values, clf, clf2_1, clf2_2)
    print("Prediction : ", out, probs)
    
    outdict = {0: 'Asymptomatic',
                1: 'Mild',
                2: 'Moderate',
                3: 'Severe' }

    ind = int(out[0])
    pred = outdict[ind]
    probs = np.round(probs[0][ind] * 100)

    if ind < 2 : 
        explainer = shap.TreeExplainer(clf2_1)
    else:
        explainer  = shap.TreeExplainer(clf2_2)
    
    shap_values = explainer.shap_values(values)
    shap.waterfall_plot(Explanation(shap_values[ind][0],explainer.expected_value[ind], feature_names=feats), show=False, max_display=10)
    plt.savefig('static/data/shap.png', dpi=600, bbox_inches='tight')
    plt.close()
    
    return render(request, 'result.html',{'pred': pred, 'probs':probs} ) 
