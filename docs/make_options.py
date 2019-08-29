from ABCD_ML.ML_Helpers import get_objects_by_type
from ABCD_ML.Metrics import get_metrics_by_type

from ABCD_ML.Models import AVALIABLE as AVALIABLE_MODELS
from ABCD_ML.Models import MODELS

from ABCD_ML.Samplers import AVALIABLE as AVALIABLE_SAMPLERS
from ABCD_ML.Samplers import SAMPLERS

from ABCD_ML.Feature_Selectors import AVALIABLE as AVALIABLE_SELECTORS
from ABCD_ML.Feature_Selectors import SELECTORS

from ABCD_ML.Ensembles import AVALIABLE as AVALIABLE_ENSEMBLES
from ABCD_ML.Ensembles import ENSEMBLES

from ABCD_ML.Default_Params import PARAMS


def get_name(obj):

    name = obj.__module__ + '.' + obj.__qualname__

    name = name.replace('.tree.tree', '.tree')
    name = name.replace('.tree.tree', '.tree')
    name = name.replace('.logistic', '')
    name = name.replace('.gpc', '')
    name = name.replace('.gpr', '')
    name = name.replace('.classification.', '.')
    name = name.replace('.regression.', '.')
    name = name.replace('.coordinate_descent.', '.')
    name = name.replace('.sklearn.', '.')
    name = name.replace('.forest.', '.')
    name = name.replace('.classes.', '.')
    name = name.replace('.base.', '.')
    name = name.replace('.multilayer_perceptron.', '.')
    name = name.replace('.univariate_selection.', '.')
    name = name.replace('.minimum_difference.', '.')
    name = name.replace('.deskl.', '.')
    name = name.replace('.exponential.', '.')
    name = name.replace('.logarithmic.', '.')
    name = name.replace('.minimum_difference.', '.')
    name = name.replace('.rrc.', '.')

    splits = name.split('.')
    for split in splits:
        if split.startswith('_'):
            name = name.replace('.' + split + '.', '.')

    return name


def get_metric_name(obj):

    name = obj.__name__
    name = name.replace('_wrapper', '')
    name = 'sklearn.metrics.' + name

    return name


def main_category(lines, name):

    stars = ''.join('*' for i in range(len(name)))
    lines.append(stars)
    lines.append(name)
    lines.append(stars)

    lines.append('')

    return lines


def add_block(lines, problem_types, AVALIABLE=None, OBJS=None):
    '''If AVALIABLE and OBJS stay none, assume that showing metrics'''

    for pt in problem_types:
        lines.append(pt)
        lines.append(''.join('=' for i in range(len(pt))))

        if AVALIABLE is None and OBJS is None:
            objs = get_metrics_by_type(pt)
            metric = True

        else:
            objs = get_objects_by_type(pt, AVALIABLE, OBJS)
            metric = False

        for obj in objs:
            lines = add_obj(lines, obj, metric=metric)

        lines.append('')

    return lines


def add_obj(lines, obj, metric=False):
    '''Obj as (obj_str, obj, obj_params),
    or if metric = True, can have just
    obj_str and obj.'''

    obj_str = obj[0]
    lines.append(obj_str)
    lines.append(''.join(['*' for i in range(len(obj_str))]))
    lines.append('')

    if metric:
        o_path = get_metric_name(obj[1])
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
        params = PARAMS[params_name].copy()
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

        if 'scipy' in str(type(value)):

            if isinstance(value.a, int):
                line += 'Random Integer Distribution ('
                line += str(value.a) + ', ' + str(value.b) + ')'

            else:
                a, b = value.interval(1)

                # Rought heuristic...
                if a == 0:
                    line += 'Random Uniform Distribution ('
                elif b/a < 11:
                    line += 'Random Uniform Distribution ('
                else:
                    line += 'Random Reciprical Distribution ('

                line += str(a) + ', ' + str(b) + ')'

        elif len(value) == 1:
            if callable(value[0]):
                line += str(value[0].__name__)
            else:
                line += str(value[0])

        elif len(value) > 50:
            line += 'Too many params to show'

        else:
            line += str(value)

        lines.append(line)
    return lines


problem_types = ['binary', 'regression', 'categorical multilabel',
                 'categorical multiclass']


lines = []

lines = main_category(lines, 'Model Types')
lines = add_block(lines, problem_types, AVALIABLE_MODELS, MODELS)

lines = main_category(lines, 'Metrics')
lines = add_block(lines, problem_types)

lines = main_category(lines, 'Samplers')
lines = add_block(lines, problem_types, AVALIABLE_SAMPLERS, SAMPLERS)

lines = main_category(lines, 'Feat Selectors')
lines = add_block(lines, problem_types, AVALIABLE_SELECTORS, SELECTORS)

lines = main_category(lines, 'Ensemble Types')
lines = add_block(lines, problem_types, AVALIABLE_ENSEMBLES, ENSEMBLES)


with open('options.rst', 'w') as f:
    for line in lines:
        f.write(line)
        f.write('\n')
