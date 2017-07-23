from Pseudo_ENTRNA import pseudo_entrna
from Pseudofree_ENTRNA import pseudofree_entrna
import RNA_Basic
import pickle
import numpy as np
import sklearn

########## pseudo-free ###############
# bp_list = [27,26,25,24,23,22,0,20,19,18,0,0,0,0,0,0,0,10,9,8,0,6,5,4,3,2,1]
# seq_str = "GGAUCCAUUCGAUUAGUGAACGGAUCC"


########## pseudoknotted ###############
bp_list = [20,19,18,17,16,0,0,0,33,0,30,0,0,0,0,5,4,3,2,1,0,0,0,0,0,0,0,0,0,11,0,0,9]
seq_str = "CUGGGUCGCAGUAACCCCAGUUAACAAAACAAG"



model_name = "LR"
if RNA_Basic.is_pseudo(bp_list) == 1:
    feature_dict = pseudo_entrna(seq_str, bp_list)
    with open("pk_" + model_name + '_clf_dict.pickle', 'rb') as f:
        clf_dict = pickle.load(f)
    with open("pk_" + model_name + '_clf_accuracy.pickle', 'rb') as f:
        clf = pickle.load(f)
else:
    feature_dict = pseudofree_entrna(seq_str, bp_list)
    with open("pf_" + model_name + '_clf_dict.pickle', 'rb') as f:
        clf_dict = pickle.load(f)
    with open("pf_" + model_name + '_clf_accuracy.pickle', 'rb') as f:
        clf = pickle.load(f)

feature_used_scaler_min = clf_dict['scaler_min'][clf_dict['best_features']]
feature_used_scaler_max = clf_dict['scaler_max'][clf_dict['best_features']]
feature_used_names = clf_dict['feature_names'][clf_dict['best_features']]
feature_fit_values = []
for ii in feature_used_names:
    feature_fit_values.append(feature_dict[ii])
feature_fit_array = np.array(feature_fit_values)
feature_fit_array = (feature_fit_array - feature_used_scaler_min) / (feature_used_scaler_max - feature_used_scaler_min).reshape(1, (len(feature_used_scaler_min)))
print clf.predict_proba(feature_fit_array)[0][1]
