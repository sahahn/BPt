import inspect


def grab_params(func):
    args = inspect.signature(func)
    args = list(args.parameters)
    args.remove('self')

    if 'kwargs' in args:
        args.remove('kwargs')

    return args


def get_mapping(lines, args):

    r = "        If \'default\', use the saved value within"
    r += " default params."

    new_lines = []
    for line in lines:
        if "If \'default\'" in line:
            new_lines.append(r)
        else:
            new_lines.append(line)
    lines = new_lines

    start_keys = ['    ' + a for a in args]
    start_lines = [[l for l in lines if l.startswith(key)][0]
                   for key in start_keys]
    start_inds = [lines.index(line) for line in start_lines]

    return_line = [l for l in lines if l.startswith('    Returns')]

    if len(return_line) > 0:
        return_ind = lines.index(return_line[0])
    else:
        return_ind = len(lines)

    start_inds.append(return_ind)

    mapping = {}
    for k in range(len(start_keys)):
        mapping[start_keys[k]] =\
            '\n'.join(lines[start_inds[k]:start_inds[k+1]])

    return mapping


def get_from_func(func):

    args = grab_params(func)
    doc = func.__doc__

    lines = doc.split('\n')
    mapping = get_mapping(lines, args)

    return doc, mapping


def get_new_docstring(o_func, r_func):

    o_doc, o_mapping = get_from_func(o_func)
    r_doc, r_mapping = get_from_func(r_func)

    for key in r_mapping:
        if key in o_mapping:
            r_doc = r_doc.replace(r_mapping[key], o_mapping[key])

    return r_doc


def get_name(obj):

    name = obj.__module__ + '.' + obj.__qualname__

    if '.<locals>.child' in name:
        name = obj.__parent_name__

    name = name.replace('.tree.tree', '.tree')
    name = name.replace('.tree.tree', '.tree')

    base_replace_list = ['logistic', 'gpc', 'gpr', 'classification',
                         'regression', 'coordinate_descent', 'sklearn',
                         'forest', 'classes', 'base', 'multilayer_perceptron',
                         'univariate_selection', 'minimum_difference', 'deskl',
                         'exponential', 'logarithmic', 'rrc', 'data',
                         'variance_threshold']

    for r in base_replace_list:
        name = name.replace('.' + r + '.', '.')

    splits = name.split('.')
    for split in splits:
        if split.startswith('_'):
            name = name.replace('.' + split + '.', '.')

    name = name.replace('BPt.extensions.Feat_Selectors.RFE_Wrapper',
                        'sklearn.feature_selection.RFE')

    return name


def get_scorer_name(obj):

    name = obj.__name__
    name = name.replace('_wrapper', '')
    name = 'sklearn.metrics.' + name

    return name
