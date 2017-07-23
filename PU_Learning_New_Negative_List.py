import numpy as np
import pickle
import pandas as pd
from sklearn import preprocessing
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import LeaveOneOut
import itertools


def PU_Learning_Pseudofree(classifier,parameter_max,version):
    df = pd.read_csv("C:/Users/congzhes/Dropbox/RNA_Slides_Words/Paper I/Data/Pseudofree/Training_New.csv",header=0)
    feature_names = df.columns.values[1:]
    data_array = np.array(df.iloc[:, 1:-1])
    scaler_min = np.min(data_array,axis = 0)
    scaler_max = np.max(data_array,axis = 0)
    min_max_scaler = preprocessing.MinMaxScaler()
    my_data = min_max_scaler.fit_transform(data_array)

    #
    # # print df['Name']
    # a_list = df[df['Name'] == a].index.tolist()
    # print a_list
    #
    # distance_matrix = distance.cdist(data_array[a_list],data_array[a_list],'euclidean')
    #
    # print distance_matrix[0]
    # print np.argmax(distance_matrix[0])
    # print a_list[np.argmax(distance_matrix[0])]


    feature_num = 9
    real_case_number = 167
    total_case_number = 1833

    clf_dict = {}
    clf_loo_dict = {}

    clf_dict['feature_names'] = feature_names
    clf_loo_dict['feature_names'] = feature_names

    best_set = []
    parameter = 0
    best_number = 0
    best_accuracy = 0
    feature_all = list(range(feature_num))

    loo_best_set = []
    loo_parameter = 0
    loo_best_number = 0
    loo_best_accuracy = 0



    for iii in range(len(feature_all)):
        for jjj in itertools.combinations(feature_all,iii+1):
            loo = LeaveOneOut(real_case_number)
            feature_set = list(jjj)
            print 'pseudofree'+classifier
            print feature_set
            data_new = my_data[:,feature_set]
            for parameter_i in range(parameter_max):
                predict_positive_number = 0
                prediction_accuracy_avg = 0
                loo_i = 0
                for train,test in loo:
                    loo_i += 1
                    data_negative = []
                    for i_train in train:
                        i_train_name = df['Name'][i_train]
                        name_list = df[df['Name'] == i_train_name].index.tolist()
                        distance_matrix = distance.cdist(data_new[name_list],data_new[name_list],'euclidean')
                        data_negative.append(data_new[name_list[np.argmax(distance_matrix[0])]])
                    data_train = data_new[train,:]  # select features used in following steps
                    data_test = data_new[test,:]
                    data_training_new = np.append(data_train,data_negative,axis = 0)  # Training Data
                    data_target_new = np.append(np.ones(real_case_number-1),np.zeros(real_case_number-1),axis = 0) # Class Label

                    if classifier == "KNN":
                        clf1 = KNeighborsClassifier(n_neighbors = parameter_i + 1)
                    elif classifier == "DT":
                        clf1 = DecisionTreeClassifier(max_depth= parameter_i +1)
                    elif classifier == "SVM_linear":
                        clf1 = SVC(kernel='linear',C=(parameter_i + 1)/2.0)
                    elif classifier == "SVM_poly":
                        clf1 = SVC(kernel='poly',C=(parameter_i + 1)/2.0)
                    elif classifier == "SVM_rbf":
                        clf1 = SVC(kernel='rbf',C=(parameter_i + 1)/2.0)
                    elif classifier == "SVM_sigmoid":
                        clf1 = SVC(kernel='sigmoid',C=(parameter_i + 1)/2.0)
                    elif classifier == "LR":
                        clf1 = LogisticRegression(C=(parameter_i + 1)/2.0)

                    clf1.fit(data_training_new,data_target_new)
                    predict_p_num = np.sum(clf1.predict(data_training_new)[:real_case_number - 1])
                    predict_n_num = real_case_number - 1 - np.sum(clf1.predict(data_training_new)[real_case_number:])
                    predict_positive_number = clf1.predict(data_test) + predict_positive_number
                    prediction_accuracy_avg = ((0.5*predict_p_num + 0.5*predict_n_num) + (loo_i - 1) * prediction_accuracy_avg) / loo_i

                if prediction_accuracy_avg > best_accuracy:
                    best_accuracy = prediction_accuracy_avg
                    best_number = predict_positive_number
                    parameter = parameter_i + 1
                    best_set = feature_set
                    clf_accuracy = clf1

                if predict_positive_number > loo_best_number:
                    loo_best_accuracy = prediction_accuracy_avg
                    loo_best_number = predict_positive_number
                    loo_parameter = parameter_i + 1
                    loo_best_set = feature_set
                    clf_loo = clf1

    clf_dict['best_features'] = best_set
    clf_dict['parameter'] = parameter
    clf_dict['best_accuracy'] = best_accuracy
    clf_dict['best_loo_number'] = best_number
    clf_dict['scaler_min'] = scaler_min
    clf_dict['scaler_max'] = scaler_max

    clf_loo_dict['best_features'] = loo_best_set
    clf_loo_dict['parameter'] = loo_parameter
    clf_loo_dict['best_accuracy'] = loo_best_accuracy
    clf_loo_dict['best_loo_number'] = loo_best_number
    clf_loo_dict['scaler_min'] = scaler_min
    clf_loo_dict['scaler_max'] = scaler_max

    with open('C:/Users/congzhes/Dropbox/RNA_Code/ENTRNA/Trained_Model/v2/'+version+'/Pseudofree/'+classifier+'_clf_accuracy.pickle', 'wb') as f1:
        pickle.dump(clf_accuracy,f1)

    with open('C:/Users/congzhes/Dropbox/RNA_Code/ENTRNA/Trained_Model/v2/'+version+'/Pseudofree/'+classifier+'_clf_dict.pickle', 'wb') as f2:
        pickle.dump(clf_dict,f2)

    with open('C:/Users/congzhes/Dropbox/RNA_Code/ENTRNA/Trained_Model/v2/'+version+'/Pseudofree/'+classifier+'_clf_loo.pickle', 'wb') as f3:
        pickle.dump(clf_loo,f3)

    with open('C:/Users/congzhes/Dropbox/RNA_Code/ENTRNA/Trained_Model/v2/'+version+'/Pseudofree/'+classifier+'_clf_loo_dict.pickle', 'wb') as f4:
        pickle.dump(clf_loo_dict,f4)


def PU_Learning_Pseudoknotted(classifier,parameter_max,version):
    df = pd.read_csv("C:/Users/congzhes/Dropbox/RNA_Slides_Words/Paper I/Data/Pseudoknotted/pseudo_New.csv",header=0)
    feature_names = df.columns.values[1:]
    data_array = np.array(df.iloc[:, 1:-1])
    scaler_min = np.min(data_array,axis = 0)
    scaler_max = np.max(data_array,axis = 0)
    min_max_scaler = preprocessing.MinMaxScaler()
    my_data = min_max_scaler.fit_transform(data_array)

    feature_num = 11
    real_case_number = 93
    total_case_number = 999

    clf_dict = {}
    clf_loo_dict = {}

    clf_dict['feature_names'] = feature_names
    clf_loo_dict['feature_names'] = feature_names

    best_set = []
    parameter = 0
    best_number = 0
    best_accuracy = 0
    feature_all = list(range(feature_num))

    loo_best_set = []
    loo_parameter = 0
    loo_best_number = 0
    loo_best_accuracy = 0

    for iii in range(len(feature_all)):
        for jjj in itertools.combinations(feature_all,iii+1):
            loo = LeaveOneOut(real_case_number)
            feature_set = list(jjj)
            data_new = my_data[:,feature_set]
            print 'pseudoknotted'+classifier
            print feature_set
            for parameter_i in range(parameter_max):
                predict_positive_number = 0
                prediction_accuracy_avg = 0
                loo_i = 0
                for train,test in loo:
                    loo_i += 1
                    data_negative = []
                    for i_train in train:
                        i_train_name = df['Name'][i_train]
                        name_list = df[df['Name'] == i_train_name].index.tolist()
                        distance_matrix = distance.cdist(data_new[name_list],data_new[name_list],'euclidean')
                        data_negative.append(data_new[name_list[np.argmax(distance_matrix[0])]])
                    data_train = data_new[train,:]  # select features used in following steps
                    data_test = data_new[test,:]
                    data_training_new = np.append(data_train,data_negative,axis = 0)  # Training Data
                    data_target_new = np.append(np.ones(real_case_number-1),np.zeros(real_case_number-1),axis = 0) # Class Label

                    if classifier == "KNN":
                        clf1 = KNeighborsClassifier(n_neighbors = parameter_i + 1)
                    elif classifier == "DT":
                        clf1 = DecisionTreeClassifier(max_depth= parameter_i +1)
                    elif classifier == "SVM_linear":
                        clf1 = SVC(kernel='linear', C=(parameter_i + 1) / 2.0)
                    elif classifier == "SVM_poly":
                        clf1 = SVC(kernel='poly', C=(parameter_i + 1) / 2.0)
                    elif classifier == "SVM_rbf":
                        clf1 = SVC(kernel='rbf', C=(parameter_i + 1) / 2.0)
                    elif classifier == "SVM_sigmoid":
                        clf1 = SVC(kernel='sigmoid', C=(parameter_i + 1) / 2.0)
                    elif classifier == "LR":
                        clf1 = LogisticRegression(C=(parameter_i + 1)/2.0)
                    clf1.fit(data_training_new, data_target_new)
                    predict_p_num = np.sum(clf1.predict(data_training_new)[:real_case_number - 1])
                    predict_n_num = real_case_number - 1 - np.sum(clf1.predict(data_training_new)[real_case_number:])
                    predict_positive_number = clf1.predict(data_test) + predict_positive_number
                    prediction_accuracy_avg = ((0.5*predict_p_num + 0.5*predict_n_num) + (loo_i - 1) * prediction_accuracy_avg) / loo_i

                if prediction_accuracy_avg > best_accuracy:
                    best_accuracy = prediction_accuracy_avg
                    best_number = predict_positive_number
                    parameter = parameter_i + 1
                    best_set = feature_set
                    clf_accuracy = clf1

                if predict_positive_number > loo_best_number:
                    loo_best_accuracy = prediction_accuracy_avg
                    loo_best_number = predict_positive_number
                    loo_parameter = parameter_i + 1
                    loo_best_set = feature_set
                    clf_loo = clf1

    clf_dict['best_features'] = best_set
    clf_dict['parameter'] = parameter
    clf_dict['best_accuracy'] = best_accuracy
    clf_dict['best_loo_number'] = best_number
    clf_dict['scaler_min'] = scaler_min
    clf_dict['scaler_max'] = scaler_max

    clf_loo_dict['best_features'] = loo_best_set
    clf_loo_dict['parameter'] = loo_parameter
    clf_loo_dict['best_accuracy'] = loo_best_accuracy
    clf_loo_dict['best_loo_number'] = loo_best_number
    clf_loo_dict['scaler_min'] = scaler_min
    clf_loo_dict['scaler_max'] = scaler_max

    with open('C:/Users/congzhes/Dropbox/RNA_Code/ENTRNA/Trained_Model/v2/'+version+'/Pseudoknot/'+classifier+'_clf_accuracy.pickle', 'wb') as f1:
        pickle.dump(clf_accuracy,f1)

    with open('C:/Users/congzhes/Dropbox/RNA_Code/ENTRNA/Trained_Model/v2/'+version+'/Pseudoknot/'+classifier+'_clf_dict.pickle', 'wb') as f2:
        pickle.dump(clf_dict,f2)

    with open('C:/Users/congzhes/Dropbox/RNA_Code/ENTRNA/Trained_Model/v2/'+version+'/Pseudoknot/'+classifier+'_clf_loo.pickle', 'wb') as f3:
        pickle.dump(clf_loo,f3)

    with open('C:/Users/congzhes/Dropbox/RNA_Code/ENTRNA/Trained_Model/v2/'+version+'/Pseudoknot/'+classifier+'_clf_loo_dict.pickle', 'wb') as f4:
        pickle.dump(clf_loo_dict,f4)


# def PU_Prediction():
#
#     with open('C:/Users/congzhes/Dropbox/RNA_Code/ENTRNA/Trained_Model/Pseudoknot/clf_accuracy.pickle', 'rb') as f1:
#         clf = pickle.load(f1)
#
#     df = pd.read_csv("C:/Users/congzhes/Dropbox/RNA_Slides_Words/Paper I/Data/Pseudoknotted/pseudo_v6_ENT8.csv",header=0)
#     data_array = np.array(df.iloc[:, :-1])
#     dim_x, dim_y = data_array.shape
#
#     min_max_scaler = preprocessing.MinMaxScaler()
#     my_data = min_max_scaler.fit_transform(data_array)
#     with open('C:/Users/congzhes/Dropbox/RNA_Code/ENTRNA/Trained_Model/Pseudoknot/clf_dict.pickle', 'rb') as f2:
#         clf_dict = pickle.load(f2)
#     for i in range(dim_x):
#         a =  (data_array[i] - clf_dict['scaler_min']) / (clf_dict['scaler_max'] - clf_dict['scaler_min'])
#         print a
#         print my_data[i]
#         print "\n"
