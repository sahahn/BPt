.. _default_pipelines:
 
**********************
Default Pipelines
**********************


You may optionally consider using one of a number of fully pre-defined pipelines. These can
be accessed though BPt.default.pipelines.

We can see a list of all available as follows:

.. ipython:: python

    import BPt as bp
    from BPt.default.pipelines import pipelines_keys
    pipelines_keys

These represent options which we can import, for example:

.. ipython:: python

    from BPt.default.pipelines import elastic_pipe
    elastic_pipe

We can go through and print each pipeline:

.. ipython:: python

    for pipeline in pipelines_keys:
        print(pipeline)
        eval(f'print(bp.default.pipelines.{pipeline})')

Note also that the individual pieces which make up the default pipelines can be accessed as well.

.. ipython:: python

    from BPt.default.pipelines import pieces_keys
    pieces_keys
    
    # Look at some
    bp.default.pipelines.u_feat
    bp.default.pipelines.svm

