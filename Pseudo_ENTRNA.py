import Vienna_FE
import RNA_Basic
import numpy as np
import math

def pseudo_entrna(seq_str,bp_list):
    feature_dict = {}
    rna_length = len(seq_str)
    bp_array = np.array(bp_list)
    bp_n = np.sum(bp_array > 0)
    bp_base, knot_seqs_list, knot_dps_list, knot_hairpins_list = RNA_Basic.pseudo_bp_decom(bp_list,seq_str)
    bp_base_array = np.array(bp_base)
    bp_base_n = np.sum(bp_base_array > 0)


    base_dp = RNA_Basic.bp_to_dp(bp_base)
    base_mfe_value = Vienna_FE.mfe_value(seq_str)
    base_fe_value = Vienna_FE.fe_value(seq_str, base_dp)

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

    bp_percent = bp_base_n / float(rna_length)
    bp_knot_percent = 1 - bp_base_n / float(bp_n)
    gc_percent = (seq_str.count('C') + seq_str.count('G')) / float(rna_length)
    rv_ent_3 = ent_3_value / max_ent_3_value
    rv_ent_4 = ent_4_value / max_ent_4_value
    rv_ent_5 = ent_5_value / max_ent_5_value
    rv_ent_6 = ent_6_value / max_ent_6_value
    rv_ent_7 = ent_7_value / max_ent_7_value
    rv_ent_8 = ent_8_value / max_ent_8_value
    knots_fe_value = RNA_Basic.knots_fe(knot_seqs_list, knot_dps_list, knot_hairpins_list)
    if base_mfe_value < 0:
        rv1 = math.fabs(base_fe_value - base_mfe_value) / math.fabs(base_mfe_value)
    else:
        rv1 = 123456
    if base_fe_value < 0:
        rv2 = math.fabs(knots_fe_value) / math.fabs(base_fe_value)
    else:
        rv2 = 456789

    # print "--------This is pseudoknotted RNA---------"
    # print "length:", rna_length
    # print "all_gc_percent:", gc_percent
    # print "all_bp_percent:", bp_percent
    # print "base_fe", base_fe_value
    # print "base_mfe", base_mfe_value
    # print "bp_knot_percent:", bp_knot_percent
    # print "bp_knots_fe", knots_fe_value
    # print "rv_ent_3:", rv_ent_3
    # print "rv_ent_4:", rv_ent_4
    # print "rv_ent_5:", rv_ent_5
    # print "rv_ent_6:", rv_ent_6
    # print "rv_ent_7:", rv_ent_7
    # print "rv_ent_8:", rv_ent_8
    feature_dict['length'] = rna_length
    feature_dict['GC_Percent'] = gc_percent
    feature_dict['BP_Per'] = bp_percent
    feature_dict['base_fe'] = base_fe_value
    feature_dict['base_mfe'] = base_mfe_value
    feature_dict['BP_P2'] = bp_knot_percent
    feature_dict['bp_knots_fe'] = knots_fe_value
    feature_dict['RV3'] = rv_ent_3
    feature_dict['RV4'] = rv_ent_4
    feature_dict['RV5'] = rv_ent_5
    feature_dict['RV6'] = rv_ent_6
    feature_dict['RV7'] = rv_ent_7
    feature_dict['RV8'] = rv_ent_8
    feature_dict['RV1'] = rv1
    feature_dict['RV2'] = rv2
    return feature_dict


