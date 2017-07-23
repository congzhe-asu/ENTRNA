from subprocess import Popen, PIPE

def fe_value(seq_list,dp_str):
    cmd = r'RNAeval.exe'
    p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=None, universal_newlines=True)
    stdout_text, stderr_text = p.communicate(input=seq_list+"\n"+dp_str+"\n\n")
    # print stdout_text
    return float(''.join(stdout_text.split('\n')[1].split(' ')[1:])[1:-1])

def mfe_value(seq_list):
    cmd = r'RNAfold.exe'
    p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=None, universal_newlines=True)
    stdout_text, stderr_text = p.communicate(input=seq_list+"\n")
    # print stdout_text
    return float(''.join(stdout_text.split('\n')[1].split(' ')[1:])[1:-1])