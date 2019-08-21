from ABCD_ML.ML_Helpers import get_objects_by_type

from ABCD_ML.Models import AVALIABLE as AVALIABLE_MODELS
from ABCD_ML.Models import MODELS

from ABCD_ML.Samplers import AVALIABLE as AVALIABLE_SAMPLERS
from ABCD_ML.Samplers import SAMPLERS

from ABCD_ML.Feature_Selectors import AVALIABLE as AVALIABLE_SELECTORS
from ABCD_ML.Feature_Selectors import SELECTORS

from ABCD_ML.Ensembles import AVALIABLE as AVALIABLE_ENSEMBLES
from ABCD_ML.Ensembles import ENSEMBLES


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


def add_block(lines, problem_types, AVALIABLE, OBJS):

    for pt in problem_types:
        lines.append(pt)
        lines.append(''.join('=' for i in range(len(pt))))

        objs = get_objects_by_type(pt, AVALIABLE, OBJS)

        for obj in objs:
            o_path = get_name(obj[1])
            lines.append('* ' + obj[0] + ' :class:`' + o_path + '`')

        lines.append('')
    return lines

problem_types = ['binary', 'regression', 'categorical multilabel',
                 'categorical multiclass']


lines = []

lines.append('***********')
lines.append('Model Types')
lines.append('***********')
lines.append('')
lines = add_block(lines, problem_types, AVALIABLE_MODELS, MODELS)

lines.append('***********')
lines.append('Samplers')
lines.append('***********')
lines.append('')
lines = add_block(lines, problem_types, AVALIABLE_SAMPLERS, SAMPLERS)

lines.append('**************')
lines.append('Feat Selectors')
lines.append('**************')
lines.append('')
lines = add_block(lines, problem_types, AVALIABLE_SELECTORS, SELECTORS)

lines.append('***************')
lines.append('Ensemble Types')
lines.append('***************')
lines.append('')
lines = add_block(lines, problem_types, AVALIABLE_ENSEMBLES, ENSEMBLES)


with open('options.rst', 'w') as f:
    for line in lines:
        f.write(line)
        f.write('\n')
