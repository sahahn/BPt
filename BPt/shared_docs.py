_shared_docs = {}

_shared_docs[
    "problem_spec"
] = """problem_spec : :class:`ProblemSpec` or 'default', optional
        This parameter accepts an instance of the
        params class :class:`ProblemSpec`.
        The ProblemSpec is essentially a wrapper
        around commonly used
        parameters needs to define the context
        the model pipeline should be evaluated in.
        It includes parameters like problem_type, scorer, n_jobs,
        random_state, etc...

        See :class:`ProblemSpec` for more information
        and for how to create an instance of this object.

        If left as 'default', then will initialize a
        ProblemSpec with default params.

        ::

            default = "default"

"""

_shared_docs[
    "problem_spec_params"
] = """problem_spec_params : :class:`ProblemSpec` params, optional
        You may also pass any valid problem spec argument-value pairs here,
        in order to override a value in the passed :class:`ProblemSpec`.
        Overriding params should be passed in kwargs style, for example:

        ::

            func(..., problem_type='binary')

    """
