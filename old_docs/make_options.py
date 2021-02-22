from BPt.helpers.ML_Helpers import get_objects_by_type, get_objects

import BPt.default.options as o
from BPt.default.default_params import PARAMS
from BPt.helpers.Docstring_Helpers import get_name, get_scorer_name


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

    lines.append('All Problem Types')
    lines.append('=================')

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
        lines.append('  Base Func Documenation: :func:`' + o_path + '`')
    else:
        o_path = get_name(obj[1])
        lines.append('  Base Class Documenation: :class:`' + o_path + '`')
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


problem_types = ['binary', 'regression', 'categorical']

lines = []

lines = main_category(lines, 'Models')
lines.append('Different base obj choices for the :class:`Model<BPt.Model>` '
             'are shown below')
lines.append('The exact str indicator, as passed to the `obj` param '
             'is represented '
             ' by the sub-heading (within "")')
lines.append('The avaliable models are further broken down by which can work' +
             'with different problem_types.')
lines.append('Additionally, a link to the original models documentation ' +
             'as well as the implemented parameter distributions are shown.')
lines.append('')
lines = add_block(lines, problem_types, o.models.AVALIABLE,
                  o.models.MODELS)

lines = main_category(lines, 'Scorers')

lines.append('Different availible choices for the `scorer` parameter' +
             ' are shown below.')
lines.append('`scorer` is accepted by ' +
             ':class:`ProblemSpec<BPt.ProblemSpec>` and ' +
             ':class:`ParamSearch<BPt.ParamSearch>`.')
lines.append('The str indicator for each `scorer` is represented by' +
             'the sub-heading (within "")')
lines.append('The avaliable scorers are further broken down by which can' +
             ' work with different problem_types.')
lines.append('Additionally, a link to the original models documentation ' +
             'is shown.')
lines.append('')

lines = add_block(lines, problem_types)


lines = main_category(lines, 'Loaders')
lines.append('Different base obj choices for the :class:`Loader<BPt.Loader>`'
             ' are shown below')
lines.append('The exact str indicator, as passed to the `obj` param '
             'is represented' +
             ' by the sub-heading (within "")')
lines.append('Additionally, a link to the original models documentation ' +
             'as well as the implemented parameter distributions are shown.')
lines.append('')
lines = add_no_type_block(lines, o.loaders.LOADERS)


lines = main_category(lines, 'Imputers')
lines.append('Different base obj choices for the :class:`Imputer<BPt.Imputer>`'
             ' are shown below')
lines.append('The exact str indicator, as passed to the `obj` '
             'param is represented' +
             ' by the sub-heading (within "")')
lines.append('Additionally, a link to the original models documentation ' +
             'as well as the implemented parameter distributions are shown.')
lines.append('Note that if the iterative imputer is requested, base_model '
             'must also be passed.')
lines.append('')
lines = add_no_type_block(lines, o.imputers.IMPUTERS)


lines = main_category(lines, 'Scalers')
lines.append('Different base obj choices for the :class:`Scaler<BPt.Scaler>` '
             'are shown below')
lines.append('The exact str indicator, as passed to the `obj` param is '
             'represented' +
             ' by the sub-heading (within "")')
lines.append('Additionally, a link to the original models documentation ' +
             'as well as the implemented parameter distributions are shown.')
lines.append('')
lines = add_no_type_block(lines, o.scalers.SCALERS)


lines = main_category(lines, 'Transformers')
lines.append('Different base obj choices for the '
             ':class:`Transformer<BPt.Transformer>` are shown below')
lines.append('The exact str indicator, as passed to the `obj` param '
             'is represented' +
             ' by the sub-heading (within "")')
lines.append('Additionally, a link to the original models documentation ' +
             'as well as the implemented parameter distributions are shown.')
lines.append('')
lines = add_no_type_block(lines, o.transformers.TRANSFORMERS)


lines = main_category(lines, 'Feat Selectors')
lines.append('Different base obj choices for the '
             ':class:`FeatSelector<BPt.FeatSelector>` are shown below')
lines.append('The exact str indicator, as passed to the `obj` '
             'param is represented' +
             ' by the sub-heading (within "")')
lines.append('The avaliable feat selectors are further broken down by '
             'which can work' +
             'with different problem_types.')
lines.append('Additionally, a link to the original models documentation ' +
             'as well as the implemented parameter distributions are shown.')
lines.append('')
lines = add_block(lines, problem_types, o.feature_selectors.AVALIABLE,
                  o.feature_selectors.SELECTORS)


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
             'as well as the implemented parameter distributions are shown.')
lines.append('Also note that ensemble may require a few extra params!')
lines.append('')
lines = add_block(lines, problem_types, o.ensembles.AVALIABLE,
                  o.ensembles.ENSEMBLES)

with open('options.rst', 'w') as f:
    for line in lines:
        f.write(line)
        f.write('\n')
