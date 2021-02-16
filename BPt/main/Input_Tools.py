

class Select(list):
    '''The Select object is an BPt specific Input Wrapper designed
    to allow hyper-parameter searches to include not
    just choice of hyper-parameter,
    but also choosing between objects
    (as well as their relevant distributions).

    Select is used to cast lists of base :class:`Model_Pipeline`
    pieces as different options.
    Consider a simple example, for specifying a selection
    between two different :class:`Models<Model>`

    ::

        model = Select([Model('linear'), Model('random forest')])

    In this example, the model passed to :class:`Model_Pipeline`
    becomes a meta object
    for selecting between the two base models.
    Note: this does require a :class:`Param_Search`
    object be passed to :class:`Model_Pipeline`.
    Notably as well, if further param distributions
    are defined within say the `Model('random forest')`,
    those will still be optimized, allowing for
    potentially even a hyper-parameter search to
    select hyper-parameter distribution...
    (i.e., if both select options had the same base model obj,
    but only differed in the passed hyper-param distributions) if one were
    so inclined...

    Other notable features of Select are, you are not limited to
    passing only two options,
    you can pass an arbitrary number... you can even,
    and I'm not even sure I want to tell you this...
    pass nested calls to Select... i.e., one of the
    Select options could be another Select, with say
    another Select...

    Lastly, explcitly note that Select is not restricted for use with Models,
    it can be used on any of
    the base class:`Model_Pipeline` piece params
    (i.e., every param but param_search and cache...).
    '''
    input_type = 'select'

    def __repr__(self):
        return 'Select(' + super().__repr__() + ')'

    def __str__(self):
        return self.__repr__()


def is_select(obj):

    try:
        if obj.input_type == 'select':
            return True
        return False

    except AttributeError:
        return False


class Duplicate(list):
    '''The Duplicate object is an BPt specific Input wrapper.
    It is designed to be cast on a list of valid scope parameters, e.g., 

    ::

        scope = Duplicate(['float', 'cat'])

    Such that the corresponding pipeline piece will be duplicated for every
    entry within Duplicate. In this case, two copies of the base object will be
    made, where both have the same remaining non-scope params
    (i.e., obj, params, extra_params),
    but one will have a scope of 'float' and the other 'cat'.

    Consider the following exentended example, where loaders is being specified
    when creating an instance of :class:`Model_Pipeline`:

    ::

        loaders = Loader(obj='identity', scope=Duplicate(['float', 'cat']))

    Is transformed in post processing / equivalent to

    ::

        loaders = [Loader(obj='identity', scope='float'),
                   Loader(obj='identity', scope='cat')]

    '''

    input_type = 'duplicate'

    def __repr__(self):
        return 'Duplicate(' + super().__repr__() + ')'

    def __str__(self):
        return self.__repr__()


def is_duplicate(obj):

    try:
        if obj.input_type == 'duplicate':
            return True
        return False

    except AttributeError:
        return False


class Pipe(list):
    '''The Pipe object is an BPt specific Input wrapper, designed
    for now to work specifically within :class:`Loader`.
    Because loader
    objects within BPt are designed to work on single files at a time,
    and further are resitricted in that they must go
    directly from some arbitrary
    file, shape and charteristics to outputted
    as a valid 2D (# Subects X # Features) array,
    it restricts potential sequential compositions.
    Pipe offers some utilty towards
    building sequential compositions.

    For example, say one had saved 4D neuroimaging fMRI timeseries,
    and they wanted
    to first employ a loader to extract timeseries by ROI
    (with say hyper-parameters defined to select which ROI to use),
    but then wanted to use
    another loader to convert the timeseries ROIs to a correlation matrix,
    and only then pass
    along the output as 1D features per subject.
    In this case, the Pipe wrapper is a greate canidate!

    Specifically, the pipe wrapper works at the level of defining a
    specific Loader, where basicially
    you are requesting that the loader you want to use be a
    Pipeline of a few different loader options,
    where the loader options are ones compatible in
    passing input to each other, e.g., the output from
    fit_transform as called on the ROI extractor is valid input
    to fit_transform of the Timeseries creator,
    and lastly the output from fit_transform of the
    Timeseries creator valid 1D feature array per subjects output.

    Consider the example in code below, where we
    assume that 'rois' is the ROI extractor,
    and 'timeseries' is the correlation matrix
    creator object (where these could be can valid loader str, or
    custom user passed objects)

    ::

        loader = Loader(obj = Pipe(['rois', 'timeseries']))

    We only passed arguments for obj above, but in our toy example
    as initially described we wanted to
    further define parameters for a parameter search across both objects.
    See below for what different options
    for passing corresponding parameter distributions are:

    ::

        # Options loader1 and loader2 tell it explicitly no params

        # Special case, if just default params = 0, will convert to 2nd case
        loader1 = Loader(obj = Pipe(['rois', 'timeseries']),
                         params = 0)

        # You can specify just a matching list
        loader2 = Loader(obj = Pipe(['rois', 'timeseries']),
                         params = [0, 0])

        # Option 3 assumes that there are pre-defined valid class param dists
        # for each of the base objects
        loader3 = Loader(obj = Pipe(['rois', 'timeseries']),
                         params = [1, 1])

        # Option 4 lets set params for the 'rois' object, w/ custom param dists
        loader4 = Loader(obj = Pipe(['rois', 'timeseries']),
                         params = [{'some custom param dist'}, 0])

    Note that still only one scope may be passed, and that scope will
    define the scope of the new combined loader.
    Also note that if extra_params is passed, the same extra_params will
    be passed when creating both individual objects.
    Where extra params behavior is to add its contents, only when the
    name of that param appears in the base classes init, s.t.
    there could exist a case where, if both 'rois' and 'timeseries'
    base objects had a parameter with the same name, passing a
    value for that name in extra params would update them
    both with the passed value.
    '''

    input_type = 'pipe'

    def __repr__(self):
        return 'Pipe(' + super().__repr__() + ')'

    def __str__(self):
        return self.__repr__()


def is_pipe(obj):

    try:
        if obj.input_type == 'pipe':
            return True
        return False

    except AttributeError:
        return False


class Value_Subset():
    ''' Value_Subset is special wrapper class for BPt designed to work with
    :ref:`Subjects` style input. As seen in :class:`Param_Search`,
    or to the `train_subjects` or `test_subjects`
    params in :func:`Evaluate <BPt.BPt_ML.Evaluate>`
    and :func:`Test <BPt.BPt_ML.Test>`.

     This wrapper can be used as follows, just specify an object as

     ::

        Value_Subset(name, values)

    Where name is the name of a loaded Strat column / feature,
    and value is the subset of values from that column to select subjects by.
    E.g., if you wanted to select just subjects of a specific sex,
    and assuming a variable was
    loaded in Strat (See :func:`Load_Strat <BPt.BPt_ML.Load_Strat>`)
    you could pass:

    ::

        subjects = Value_Subset('sex', 0)

    Which would specify only subjects with 'sex' equal to 0.
    You may also optionally pass more than one value to values, E.g.,

    ::

        subjects = Values_Subset(name='site', values=[0, 1, 5])

    Would select the subset of subjects from sites 0, 1 and 5.

    There is one more parameter which represents if encoded values should be
    used or not. What this asks is that should the actual ordinal post encoded
    value be specified, or the should the value be set based on the original
    encoded name. For example, let's say sex originally had values 'M' and 'F'
    and then binarize was used to set it to 0 and 1. If decode_values is set
    as the default value of False, then you must pass value = 0 or 1, but
    if decode_values = True, then you must pass value = 'M' or 'F'. E.g.,

    ::

        subjects = Value_Subset('sex', 'M', decode_values=True)

    '''

    input_type = 'value_subset'

    def __init__(self, name, values, decode_values=False):

        self.name = name
        self.values = values
        self.decode_values = decode_values

        if isinstance(self.name, list):
            raise ValueError('name cannot be list / array-like!')

        if not isinstance(self.decode_values, bool):
            raise ValueError('decode_values must be a bool')

    def __repr__(self):
        return 'Value_Subset(name=' + str(self.name) + ', value=' + \
          str(self.value) + ', decode_values=' + \
          str(self.decode_values) + ')'

    def __str__(self):
        return self.__repr__()


def is_value_subset(obj):

    try:
        if obj.input_type == 'value_subset':
            return True
        return False

    except AttributeError:
        return False


class Intersection(list):
    '''The Intersection class is a special
    wrapper class used to request the intersection
    of two valid arguments for :ref:`Subjects`.

    For example:

    ::

        subjects = Intersection(['train',
                                [1, 2, 3]])

    Would specify subjects as the intersection of the loaded
    train subjects and subjects 1, 2 and 3.
    '''

    input_type = 'intersection'

    def __repr__(self):
        return 'Intersection(' + super().__repr__() + ')'

    def __str__(self):
        return self.__repr__()


def is_special(obj):
    return hasattr(obj, 'input_type')
