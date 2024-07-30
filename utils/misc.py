import argparse

def update_dict_by_missing_keys(dic1, dic2):
    """
    Add key-value pairs to dic1 if the key exists in dic2, but not dic1. 
    Use case: dic2 is some new config with additional key-value pairs.
    """
    out = {}
    for k in dic2:
        if k not in dic1:
            out[k] = dic2[k]
        else:
            out[k] = dic1[k]
    return out

def updata_ns_by_missing_keys(ns1, ns2):
    return dic2ns(update_dict_by_missing_keys( ns2dic(ns1), ns2dic(ns2)))

def dic2ns(dic):
    return argparse.Namespace(**dic)

def ns2dic(ns):
    return vars(ns)

def test_update_dict_by_missing_keys():
    dic1 = {
        'a': 1
    }

    dic2 = {
        'a': 2,
        'b': 3
    }
    out = update_dict_by_missing_keys(dic1, dic2)
    print(out)
    assert out['a'] == 1 and out['b'] == 3
    print('>> PASSED')

if __name__ == '__main__':
    """
    python -m utils.misc
    """
    test_update_dict_by_missing_keys()