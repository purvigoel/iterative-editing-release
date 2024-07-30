import torch
import roma

def apply_rotvec_to_aa(rotvec, aa):
    N = aa.shape[0]
    rotvec = rotvec.repeat(N, 1)
    return roma.rotvec_composition([rotvec, aa])


def apply_rotvec_to_aa2(rotvec, aa):
    N = aa.shape[0]
    rotvec = rotvec.repeat(N, 1)
    return roma.rotvec_composition([aa, rotvec])