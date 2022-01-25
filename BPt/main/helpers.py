def clean_str(in_str):

    # If float input, want to
    # represent without decimals if
    # they are just 0's
    if isinstance(in_str, float):
        as_int_str = f'{in_str:.0f}'
        if float(as_int_str) == in_str:
            in_str = as_int_str

    # Make sure str
    in_str = str(in_str)

    # Get rid of some common repr issues
    in_str = in_str.replace('"', '')
    in_str = in_str.replace("'", '')

    return in_str
