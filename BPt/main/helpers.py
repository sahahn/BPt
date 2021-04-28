def clean_str(in_str):

    in_str = str(in_str)
    in_str = in_str.replace('"', '')
    in_str = in_str.replace("'", '')

    return in_str
