from .main.input_operations import BPtInputMixIn


def is_array_like(in_val):

    if hasattr(in_val, '__len__') and (not isinstance(in_val, str)) and \
     (not isinstance(in_val, dict)) and (not hasattr(in_val, 'fit')) and \
     (not hasattr(in_val, 'transform')):
        return True
    else:
        return False


def conv_to_list(in_val, amt=1):

    if in_val is None:
        return None

    if not is_array_like(in_val) or isinstance(in_val, BPtInputMixIn):
        in_val = [in_val for i in range(amt)]

    return in_val
