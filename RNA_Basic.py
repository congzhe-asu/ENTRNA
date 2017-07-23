import math
import numpy as np
import Vienna_FE


def is_pseudo(bp_list):
    bp_start = []
    nn = len(bp_list)
    for i in range(nn-1):
        if i+1 < int(bp_list[i]):
            bp_start.append(i+1)
    flag = 0
    bp_n = len(bp_start)
    for i in range(bp_n):
        for j in range(bp_n-i-1):
            if (bp_start[i] < bp_start[i+1+j] and int(bp_list[bp_start[i]-1]) > int(bp_list[bp_start[i+1+j]-1])) or (int(bp_list[bp_start[i]-1]) < bp_start[i+1+j]):
                continue
            else:
                flag = 1        # Pseudoknotted
                break
    return flag

        
def entropy(seq, mm):
    seqDict = {}
    n = len(seq)
    for i in range(n-mm+1):
        if seq[i:i+mm] in seqDict:
            seqDict[seq[i:i+mm]] += 1
        else:
            seqDict[seq[i:i+mm]] = 1
    nn = float(n-mm+1)
    x = 0
    for i in seqDict:
        prob = seqDict[i]/nn
        seqDict[i] = - prob * math.log(prob, 2)
        x += seqDict[i]
    return x


def entropy_max(nn, ent_n):
    entropy_max_value = 0
    if nn <= ent_n:
        entropy_max_value = 0
    elif nn - ent_n + 1 <= 4**ent_n:
            prob = 1/float(nn - ent_n + 1)
            for i in range(nn - ent_n + 1):
                entropy_max_value +=  (-(prob) * math.log(prob, 2))
    elif nn - ent_n + 1 > 4**ent_n:
        a = (nn - ent_n + 1) / (4**ent_n)
        b = (nn - ent_n + 1) % (4**ent_n)
        entropy_max_value = (-b) * (a+1)/float(nn - ent_n + 1) * math.log((a+1)/float(nn - ent_n + 1), 2) - ((4**ent_n-b) * a / float(nn - ent_n + 1) * math.log(a/float(nn - ent_n + 1), 2))
    return entropy_max_value


def bp_to_dp(bp_list):
    dp_str = ""
    for i in range(len(bp_list)):
        if bp_list[i] == 0:
            dp_str += "."
        elif i + 1 < bp_list[i]:
            dp_str += "("
        elif i + 1 > bp_list[i]:
            dp_str += ")"
        elif i + 1 == bp_list[i]:
            print i
            print "ERROR!!!!!!!!!!!!!!!"
    return dp_str


def HasBranch(DP):
    left_start = 0
    right_end = len(DP)
    for i in range(len(DP)):
        if DP[i] == "(":
            if i > left_start:
                left_start = i
        elif DP[i] == ")":
            if i < right_end:
                right_end = i
    if left_start < right_end:
        return 0
    else:
        return 1


def UnpairSegNum(segAssign):
####################################################
# 0: keep unpair
# 1: keep bp
# 2: all deleted
####################################################
    length = len(segAssign)
    segUnpair = np.zeros(length,dtype = int)
    Main_Start = np.amin(np.where(segAssign == 1))
    Main_End = np.amax(np.where(segAssign == 1))
    Sub_Start = np.amin(np.where(segAssign == 2))
    Sub_End = np.amax(np.where(segAssign == 2))
    for i in range(length):
        if segAssign[i] == 2:
            segUnpair[i] = 2
        elif segAssign[i] == 1:
            segUnpair[i] = 1
        elif segAssign[i] == 0:
            if i < Main_End and i > Main_Start:
                segUnpair[i] = 0
            elif i > Main_End and Main_End > Sub_End:
                segUnpair[i] = 2
            elif i < Main_Start and Main_Start < Sub_Start:
                segUnpair[i] = 2
            else:
                segUnpair[i] = 2
    return segUnpair


def DeleteLeft(ii,segBP,arm,seg_i):
    rna_length = len(segBP)
    for i in range(rna_length):
        if i > ii and arm[i] == seg_i:
            segBP[i] = segBP[i] - 1
    return segBP


def DeleteMiddle(ii,segBP,fArray,arm,seg_i):
    rna_length = len(segBP)
    for i in range(rna_length):
        if fArray[i] - 1 > ii and arm[i] == seg_i:
            segBP[i] = segBP[i] - 1
    return segBP


def ExtractSEQ(segBP,Seq):
    segSEQ = ""
    for i in range(len(segBP)):
        if segBP[i] >= 0:
            segSEQ += Seq[i]
    return segSEQ


def DeleteRight(ii,segBP,arm,seg_i):
    return segBP


def InsertAAAAA(segSEQFinal_Before,segDPFinal_Before):
    Insert_Start = 0
    Insert_End = len(segDPFinal_Before)
    for i in range(len(segDPFinal_Before)):
        if segDPFinal_Before[i] == "(":
            if i > Insert_Start:
                Insert_Start = i
        elif segDPFinal_Before[i] == ")":
            if i < Insert_End:
                Insert_End = i
    segSEQFinal =  segSEQFinal_Before[:Insert_Start+1]+"AAAAA"+segSEQFinal_Before[Insert_End:]
    segDPFinal = segDPFinal_Before[:Insert_Start+1]+"....."+segDPFinal_Before[Insert_End:]
    HairpinSeq = segSEQFinal_Before[Insert_Start]+"AAAAA"+segSEQFinal_Before[Insert_End]
    return HairpinSeq,segSEQFinal, segDPFinal


def pseudo_bp_decom(bp_list, seq_str):
    bp_array = np.asarray(bp_list, dtype=int)
    arm = np.ones(len(bp_array), dtype=int) * -1
    armnum = 1
#####################################################################
    for nn in range(len(bp_array)):
        if arm[nn] >= 0:
            continue
        elif bp_array[nn] == 0:
            arm[nn] = 0
        elif nn == 0:
            arm[nn] = armnum
            arm[bp_array[nn] - 1] = armnum
        else:
            if bp_array[nn] == bp_array[nn - 1] - 1:
                arm[nn] = armnum
                arm[bp_array[nn] - 1] = armnum
            else:
                armnum += 1
                arm[nn] = armnum
                arm[bp_array[nn] - 1] = armnum
##################### finish small segment assignment##################


#####################################################################
    unique, counts = np.unique(arm, return_counts=True)
    freq = dict(zip(unique, counts))
    freq_list = sorted(freq, key = freq.get, reverse=True)
############# arm frequency list after sorting########################

########### combine arms into segments ###############
    segnum_n = 1
    seg_list = []
    seg_array = np.zeros(len(bp_array),dtype = int)
############### seg global info initialization ########################


#####################  Primary Chain############################
    seg_list.append(segnum_n)
    bpknotfreemain_array = np.zeros(len(bp_array), dtype=int)
    for i_1 in range(len(freq_list)):
        if freq_list[i_1] == 0:
            continue
        else:
            if is_pseudo(bpknotfreemain_array + bp_array * (arm == freq_list[i_1])) == 0: # 0 means no pseudo
                bpknotfreemain_array += bp_array * (arm == freq_list[i_1])
                seg_array += segnum_n * (arm == freq_list[i_1])
                freq_list[i_1] = 0
###############################################################

    segSEQ_arrays_list = []
    segDP_strs_list = []
    Hairpin_strs_list = []

######################  Decompose Knotted Area ##########################
    while sum(freq_list) > 0:
        bpknotfree_array = np.zeros(len(bp_array), dtype=int)
        segnum_n += 1
        seg_list.append(segnum_n)
        for i_22 in range(len(freq_list)):
            if freq_list[i_22] > 0:
                bpknotfree_array += bp_array * (arm == freq_list[i_22])
                seg_array += segnum_n * (arm == freq_list[i_22])
                freq_list[i_22] = 0
                break       ### pick one arm as candidate and combine pseudofree arms

        for i_2 in range(len(freq_list)):
            if freq_list[i_2] > 0 and is_pseudo(bpknotfree_array + bp_array * (arm == freq_list[i_2])) == 0:
                temp_dp = bp_to_dp(bpknotfree_array + bp_array * (arm == freq_list[i_2]))
                left_start = temp_dp.find("(")
                left_end = len(bp_array)
                right_start = temp_dp.find(")")
                right_end = 0
                for ii in range(len(bp_array)):
                    if temp_dp[ii] == "(":
                        left_end = ii
                    elif temp_dp[ii] == ")":
                        right_end = ii

                if HasBranch(temp_dp) == 0 and sum(bpknotfreemain_array[left_start:left_end+1]) == 0 and sum(bpknotfreemain_array[right_start:right_end+1]) == 0:
                    bpknotfree_array += bp_array * (arm == freq_list[i_2])
                    seg_array += segnum_n * (arm == freq_list[i_2])
                    freq_list[i_2] = 0

    for i_3 in seg_list:
        if i_3 > 1:
            segNew = 2*(seg_array != i_3)-2*(seg_array == 0)+1*(seg_array == i_3)
            ####################################################
            # 0: Unpaired
            # 1: Main
            # 2: Paired but not in this structure
            ####################################################
            segUnpair = UnpairSegNum(segNew)
            segArm = np.delete(np.unique(arm*(seg_array == i_3)),0)
            segBP = np.copy(bp_array)
            rna_length = len(bp_array)
            for seg_i in segArm:
                seg_Start = np.amin(np.where(arm == seg_i))
                seg_End = np.amax(np.where(arm == seg_i))
                for ii in range(rna_length):
                   if segUnpair[ii] == 2:
                        segBP[ii] = -10
                        if ii < seg_Start:
                            segBP = DeleteLeft(ii,segBP,arm,seg_i)
                        elif seg_Start < ii and ii < seg_End:
                            segBP = DeleteMiddle(ii,segBP,bp_array,arm,seg_i)
                        elif ii > seg_End:
                            segBP = DeleteRight(ii,segBP,arm,seg_i)
            idxs = (segBP != -10)
            segBPFinal = segBP[idxs]
            segSEQFinal_Before = ExtractSEQ(segBP,seq_str)
            segDPFinal_Before = bp_to_dp(segBPFinal)
            HairpinSeq,segSEQFinal,segDPFinal = InsertAAAAA(segSEQFinal_Before,segDPFinal_Before)

            segSEQ_arrays_list.append(segSEQFinal)
            segDP_strs_list.append(segDPFinal)
            Hairpin_strs_list.append(HairpinSeq)

    return bpknotfreemain_array,segSEQ_arrays_list,segDP_strs_list,Hairpin_strs_list


def knots_fe(knot_seqs_list, knot_dps_list, knot_hairpins_list):
    seg_n = len(knot_hairpins_list)
    knots_fe_value = 0
    hairpins_fe = 0
    dp_hairpin = "(.....)"
    for i in range(seg_n):
        if knot_dps_list[i].count("(") + knot_dps_list[i].count(")") > 2:
            knots_fe_value += Vienna_FE.fe_value(knot_seqs_list[i],knot_dps_list[i])
            hairpins_fe += Vienna_FE.fe_value(knot_hairpins_list[i],dp_hairpin)

    knots_fe_value -= hairpins_fe
    return knots_fe_value




