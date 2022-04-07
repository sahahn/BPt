import BPt.default.options as o
from BPt.default.params.default_params import PARAMS
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
    start_lines = [[ln for ln in lines if ln.startswith(key)][0]
                   for key in start_keys]
    start_inds = [lines.index(line) for line in start_lines]

    return_line = [ln for ln in lines if ln.startswith('    Returns')]

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
                         'variance_threshold', 'nifti_labels_masker']

    for r in base_replace_list:
        name = name.replace('.' + r + '.', '.')

    splits = name.split('.')
    for split in splits:
        if split.startswith('_'):
            name = name.replace('.' + split + '.', '.')

    return name


def get_scorer_name(obj):

    name = obj.__name__
    name = name.replace('_wrapper', '')
    name = 'sklearn.metrics.' + name

    return name


def get_avaliable_by_type(AVALIABLE):

    avaliable_by_type = {}

    for pt in AVALIABLE:

        avaliable_by_type[pt] = set()
        for select in AVALIABLE[pt]:
            avaliable_by_type[pt].add(AVALIABLE[pt][select])

        avaliable_by_type[pt] = list(avaliable_by_type[pt])
        avaliable_by_type[pt].sort()

    return avaliable_by_type


def get_objects_by_type(problem_type, AVALIABLE=None, OBJS=None):

    avaliable_by_type = get_avaliable_by_type(AVALIABLE)

    objs = []
    for obj_str in avaliable_by_type[problem_type]:

        if 'basic ensemble' not in obj_str:
            obj = OBJS[obj_str][0]
            obj_params = OBJS[obj_str][1]
            objs.append((obj_str, obj, obj_params))

    return objs


def get_objects(OBJS):

    objs = []
    for obj_str in OBJS:

        obj = OBJS[obj_str][0]
        obj_params = OBJS[obj_str][1]
        objs.append((obj_str, obj, obj_params))

    return objs


def main_category(lines, name):

    lines.append('.. _' + name + ':')
    lines.append(' ')

    stars = ''.join('*' for i in range(len(name)))
    lines.append(stars)
    lines.append(name)
    lines.append(stars)

    lines.append('')

    return lines


def add_block(lines, problem_types, AVALIABLE=None, OBJS=None):
    '''If AVALIABLE and OBJS stay none, assume that showing scorers'''

    for pt in problem_types:
        lines.append(pt)
        lines.append(''.join('=' for i in range(len(pt))))

        if AVALIABLE is None and OBJS is None:
            objs = o.scorers.get_scorers_by_type(pt)
            scorer = True

        else:
            objs = get_objects_by_type(pt, AVALIABLE, OBJS)
            scorer = False

        for obj in objs:
            lines = add_obj(lines, obj, scorer=scorer)

        lines.append('')

    return lines


def add_no_type_block(lines, OBJS):

    # lines.append('All Problem Types')
    # lines.append('=================')

    objs = get_objects(OBJS)

    for obj in objs:
        lines = add_obj(lines, obj, scorer=False)

    lines.append('')
    return lines


def add_obj(lines, obj, scorer=False):
    '''Obj as (obj_str, obj, obj_params),
    or if scorer = True, can have just
    obj_str and obj.'''

    obj_str = '"' + obj[0] + '"'
    lines.append(obj_str)
    lines.append(''.join(['*' for i in range(len(obj_str))]))
    lines.append('')

    if scorer:
        o_path = get_scorer_name(obj[1])
        lines.append('  Base Func Documentation: :func:`' + o_path + '`')
    else:
        o_path = get_name(obj[1])
        lines.append('  Base Class Documentation: :class:`' + o_path + '`')
        lines = add_params(lines, obj[2])

    lines.append('')
    return lines


def add_params(lines, obj_params):

    lines.append('')
    lines.append('  Param Distributions')
    lines.append('')

    for p in range(len(obj_params)):

        # Get name
        params_name = obj_params[p]
        lines.append('\t' + str(p) + '. "' + params_name + '" ::')
        lines.append('')

        # Show info on the params
        params = PARAMS[params_name]
        if len(params) > 0:
            lines = add_param(lines, params)
        else:
            lines.append('\t\tdefaults only')

        lines.append('')

    return lines


def add_param(lines, params):

    for key in params:

        line = '\t\t' + key + ': '
        value = params[key]
        line += repr(value)
        lines.append(line)

    return lines


def save(lines, name):

    with open('source/options/pipeline_options/' + name + '.rst', 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')


def save_all():

    problem_types = ['binary', 'regression', 'categorical']

    lines = []
    lines = main_category(lines, 'Models')
    lines.append('Different base obj choices for the'
                 ' :class:`Model<BPt.Model>` '
                 'are shown below')
    lines.append('The exact str indicator, as passed to the `obj` param '
                 'is represented '
                 ' by the sub-heading (within "")')
    lines.append('The avaliable models are further broken down by '
                 'which can work ' +
                 'with different problem_types.')
    lines.append('Additionally, a link to the original models documentation ' +
                 'as well as the implemented parameter '
                 'distributions are shown.')
    lines.append('')
    lines = add_block(lines, problem_types, o.models.AVALIABLE,
                      o.models.MODELS)
    save(lines, 'models')

    lines = []
    lines = main_category(lines, 'Scorers')
    lines.append('Different available choices for the `scorer` parameter' +
                 ' are shown below.')
    lines.append('`scorer` is accepted by ' +
                 ':class:`ProblemSpec<BPt.ProblemSpec>` and ' +
                 ':class:`ParamSearch<BPt.ParamSearch>`.')
    lines.append('The str indicator for each `scorer` is represented by' +
                 ' the sub-heading (within "")')
    lines.append('The avaliable scorers are further broken down by which can' +
                 ' work with different problem_types.')
    lines.append('Additionally, a link to the original models documentation ' +
                 'is shown.')
    lines.append('')
    lines = add_block(lines, problem_types)
    save(lines, 'scorers')

    lines = []
    lines = main_category(lines, 'Loaders')
    lines.append('Different base obj choices for the '
                 ':class:`Loader<BPt.Loader>`'
                 ' are shown below')
    lines.append('The exact str indicator, as passed to the `obj` param '
                 'is represented' +
                 ' by the sub-heading (within "")')
    lines.append('Additionally, a link to the original models documentation ' +
                 'as well as the implemented parameter '
                 'distributions are shown.')
    lines.append('')
    lines = add_no_type_block(lines, o.loaders.LOADERS)
    save(lines, 'loaders')

    lines = []
    lines = main_category(lines, 'Imputers')
    lines.append('Different base obj choices for the '
                 ':class:`Imputer<BPt.Imputer>`'
                 ' are shown below')
    lines.append('The exact str indicator, as passed to the `obj` '
                 'param is represented' +
                 ' by the sub-heading (within "")')
    lines.append('Additionally, a link to the original models documentation ' +
                 'as well as the implemented parameter '
                 'distributions are shown.')
    lines.append('Note that if the iterative imputer is requested, base_model '
                 'must also be passed.')
    lines.append('')
    lines = add_no_type_block(lines, o.imputers.IMPUTERS)
    save(lines, 'imputers')

    lines = []
    lines = main_category(lines, 'Scalers')
    lines.append('Different base obj choices for the '
                 ':class:`Scaler<BPt.Scaler>` '
                 'are shown below')
    lines.append('The exact str indicator, as passed to the `obj` param is '
                 'represented' +
                 ' by the sub-heading (within "")')
    lines.append('Additionally, a link to the original models documentation ' +
                 'as well as the implemented parameter '
                 'distributions are shown.')
    lines.append('')
    lines = add_no_type_block(lines, o.scalers.SCALERS)
    save(lines, 'scalers')

    lines = []
    lines = main_category(lines, 'Samplers')
    lines.append('Different base obj choices for the '
                 ':class:`Scaler<BPt.Sampler>` '
                 'are shown below')
    lines.append('The exact str indicator, as passed to the `obj` param is '
                 'represented' +
                 ' by the sub-heading (within "")')
    lines.append('Additionally, a link to the original models documentation ' +
                 'as well as the implemented parameter '
                 'distributions are shown.')
    lines.append('')
    lines = add_no_type_block(lines, o.scalers.SCALERS)
    save(lines, 'samplers')

    lines = []
    lines = main_category(lines, 'Transformers')
    lines.append('Different base obj choices for the '
                 ':class:`Transformer<BPt.Transformer>` are shown below')
    lines.append('The exact str indicator, as passed to the `obj` param '
                 'is represented' +
                 ' by the sub-heading (within "")')
    lines.append('Additionally, a link to the original models documentation ' +
                 'as well as the implemented parameter '
                 'distributions are shown.')
    lines.append('')
    lines = add_no_type_block(lines, o.transformers.TRANSFORMERS)
    save(lines, 'transformers')

    lines = []
    lines = main_category(lines, 'Feat Selectors')
    lines.append('Different base obj choices for the '
                 ':class:`FeatSelector<BPt.FeatSelector>` are shown below')
    lines.append('The exact str indicator, as passed to the `obj` '
                 'param is represented' +
                 ' by the sub-heading (within "")')
    lines.append('The avaliable feat selectors are further broken down by '
                 'which can work '
                 'with different problem_types.')
    lines.append('Additionally, a link to the original models documentation ' +
                 'as well as the implemented parameter '
                 'distributions are shown.')
    lines.append('')
    lines = add_block(lines, problem_types, o.feature_selectors.AVALIABLE,
                      o.feature_selectors.SELECTORS)
    save(lines, 'selectors')

    lines = []
    lines = main_category(lines, 'Ensemble Types')
    lines.append('Different base obj choices for the '
                 ':class:`Ensemble<BPt.Ensemble>` are shown below')
    lines.append('The exact str indicator, as passed to the `obj` '
                 'param is represented' +
                 ' by the sub-heading (within "")')
    lines.append('The avaliable ensembles are further broken down by ' +
                 'which can work' +
                 'with different problem_types.')
    lines.append('Additionally, a link to the original models documentation ' +
                 'as well as the implemented parameter'
                 ' distributions are shown.')
    lines.append('Also note that ensemble may require a few extra params!')
    lines.append('')
    lines = add_block(lines, problem_types, o.ensembles.AVALIABLE,
                      o.ensembles.ENSEMBLES)
    save(lines, 'ensembles')


if __name__ == '__main__':
    save_all()
