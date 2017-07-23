import Vienna_FE
import RNA_Basic
import math
# import RNA


def pseudofree_entrna_ubuntu(seq_str, bp_list):
    feature_dict = {}
    rna_length = len(seq_str)
    dp_str = RNA_Basic.bp_to_dp(bp_list)
    # a = RNA.fold_compound(seq_str)
    # (s,mfe_value) = a.mfe()
    # fe_value = a.eval_structure(dp_str)
    mfe_value = Vienna_FE.mfe_value(seq_str)
    fe_value = Vienna_FE.fe_value(seq_str, base_dp)
    ent_3_value = RNA_Basic.entropy(seq_str, 3)
    ent_4_value = RNA_Basic.entropy(seq_str, 4)
    ent_5_value = RNA_Basic.entropy(seq_str, 5)
    ent_6_value = RNA_Basic.entropy(seq_str, 6)
    ent_7_value = RNA_Basic.entropy(seq_str, 7)
    ent_8_value = RNA_Basic.entropy(seq_str, 8)
    max_ent_3_value = RNA_Basic.entropy_max(rna_length, 3)
    max_ent_4_value = RNA_Basic.entropy_max(rna_length, 4)
    max_ent_5_value = RNA_Basic.entropy_max(rna_length, 5)
    max_ent_6_value = RNA_Basic.entropy_max(rna_length, 6)
    max_ent_7_value = RNA_Basic.entropy_max(rna_length, 7)
    max_ent_8_value = RNA_Basic.entropy_max(rna_length, 8)

    gc_percent = (seq_str.count('C') + seq_str.count('G')) / float(rna_length)
    bp_percent = 2*dp_str.count('(')/float(rna_length)
    if mfe_value == 0:
        rv_fe = 100000
    else:
        rv_fe = math.fabs(fe_value - mfe_value) / math.fabs(mfe_value)
    rv_ent_3 = ent_3_value / max_ent_3_value
    rv_ent_4 = ent_4_value / max_ent_4_value
    rv_ent_5 = ent_5_value / max_ent_5_value
    rv_ent_6 = ent_6_value / max_ent_6_value
    rv_ent_7 = ent_7_value / max_ent_7_value
    rv_ent_8 = ent_8_value / max_ent_8_value

    # print "-------This is a pseudoknot-free RNA--------"
    # print "length:", rna_length
    # print "gc_percent:", gc_percent
    # print "bp_percent:", bp_percent
    # print "fe", fe_value
    # print "mfe", mfe_value
    # print "rv_fe:", rv_fe
    # print "rv_ent_3:", rv_ent_3
    # print "rv_ent_4:", rv_ent_4
    # print "rv_ent_5:", rv_ent_5
    # print "rv_ent_6:", rv_ent_6
    # print "rv_ent_7:", rv_ent_7
    # print "rv_ent_8:", rv_ent_8

    feature_dict['length'] = rna_length
    feature_dict['GC_Percent'] = gc_percent
    feature_dict['BP_Percent'] = bp_percent
    feature_dict['fe'] = fe_value
    feature_dict['mfe'] = mfe_value
    feature_dict['RVFE'] = rv_fe
    feature_dict['RV3'] = rv_ent_3
    feature_dict['RV4'] = rv_ent_4
    feature_dict['RV5'] = rv_ent_5
    feature_dict['RV6'] = rv_ent_6
    feature_dict['RV7'] = rv_ent_7
    feature_dict['RV8'] = rv_ent_8
    return feature_dict



def pseudofree_entrna(seq_str, bp_list):
    feature_dict = {}
    rna_length = len(seq_str)
    dp_str = RNA_Basic.bp_to_dp(bp_list)
    fe_value = Vienna_FE.fe_value(seq_str, dp_str)
    mfe_value = Vienna_FE.mfe_value(seq_str)
    ent_3_value = RNA_Basic.entropy(seq_str, 3)
    ent_4_value = RNA_Basic.entropy(seq_str, 4)
    ent_5_value = RNA_Basic.entropy(seq_str, 5)
    ent_6_value = RNA_Basic.entropy(seq_str, 6)
    ent_7_value = RNA_Basic.entropy(seq_str, 7)
    ent_8_value = RNA_Basic.entropy(seq_str, 8)
    max_ent_3_value = RNA_Basic.entropy_max(rna_length, 3)
    max_ent_4_value = RNA_Basic.entropy_max(rna_length, 4)
    max_ent_5_value = RNA_Basic.entropy_max(rna_length, 5)
    max_ent_6_value = RNA_Basic.entropy_max(rna_length, 6)
    max_ent_7_value = RNA_Basic.entropy_max(rna_length, 7)
    max_ent_8_value = RNA_Basic.entropy_max(rna_length, 8)

    gc_percent = (seq_str.count('C') + seq_str.count('G')) / float(rna_length)
    bp_percent = 2*dp_str.count('(')/float(rna_length)
    if mfe_value == 0:
        rv_fe = 100000
    else:
        rv_fe = math.fabs(fe_value - mfe_value) / math.fabs(mfe_value)
    rv_ent_3 = ent_3_value / max_ent_3_value
    rv_ent_4 = ent_4_value / max_ent_4_value
    rv_ent_5 = ent_5_value / max_ent_5_value
    rv_ent_6 = ent_6_value / max_ent_6_value
    rv_ent_7 = ent_7_value / max_ent_7_value
    rv_ent_8 = ent_8_value / max_ent_8_value

    # print "-------This is a pseudoknot-free RNA--------"
    # print "length:", rna_length
    # print "gc_percent:", gc_percent
    # print "bp_percent:", bp_percent
    # print "fe", fe_value
    # print "mfe", mfe_value
    # print "rv_fe:", rv_fe
    # print "rv_ent_3:", rv_ent_3
    # print "rv_ent_4:", rv_ent_4
    # print "rv_ent_5:", rv_ent_5
    # print "rv_ent_6:", rv_ent_6
    # print "rv_ent_7:", rv_ent_7
    # print "rv_ent_8:", rv_ent_8

    feature_dict['length'] = rna_length
    feature_dict['GC_Percent'] = gc_percent
    feature_dict['BP_Percent'] = bp_percent
    feature_dict['fe'] = fe_value
    feature_dict['mfe'] = mfe_value
    feature_dict['RVFE'] = rv_fe
    feature_dict['RV3'] = rv_ent_3
    feature_dict['RV4'] = rv_ent_4
    feature_dict['RV5'] = rv_ent_5
    feature_dict['RV6'] = rv_ent_6
    feature_dict['RV7'] = rv_ent_7
    feature_dict['RV8'] = rv_ent_8
    return feature_dict

