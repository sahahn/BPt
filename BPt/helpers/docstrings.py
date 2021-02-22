

ps_doc =\
    '''
    problem_spec : :class:`ProblemSpec` or 'default', optional
        `problem_spec` accepts an instance of the
        params class :class:`ProblemSpec`.
        This object is essentially a wrapper around commonly used
        parameters needs to define the context
        the model pipeline should be evaluated in.
        It includes parameters like problem_type, scorer, n_jobs,
        random_state, etc...
        See :class:`ProblemSpec` for more information
        and for how to create an instance of this object.

        If left as 'default', then will initialize a
        ProblemSpec with default params.

        ::

            default = 'default'

    '''
