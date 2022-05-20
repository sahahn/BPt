.. _pipeline_objects:

{{ header }}

*******************
Pipeline Objects
*******************

.. currentmodule:: BPt

The pipeline classes :class:`Pipeline` and :class:`ModelPipeline`
are made up of base pipeline objects / pieces. These are based on the
scikit learn concept of a :class:`Pipeline <sklearn.pipeline.Pipeline>`.

Across all base :class:`ModelPipeline<BPt.ModelPipeline>` pieces,
e.g., :class:`Model<BPt.Model>` or :class:`Scaler<BPt.Scaler>`,
there exists an `obj` param when initializing these objects. 
Beyond choice of `obj` parameter, pipeline objects share a number of other common
parameters which allow for extra customization, as well as some objects which have unique
parameters. Shared parameters include :ref:`params<params>`, :ref:`scope<scope>`, :ref:`extra_params<extra_params>`> and cache_loc.

obj
~~~~~~

The 'obj' parameter is the core parameter for any pipeline object. It
can broadly refer to either a str, 
which indicates a valid pre-defined custom obj for that piece, or depending
on the pieces, this parameter can be passed a custom object directly.

For example if we want to make an 
instance of :class:`RobustScaler <sklearn.preprocessing.RobustScaler>` from sklearn:

.. ipython:: python

    import BPt as bp
    scaler = bp.Scaler('robust', scope='all')
    scaler

    # See what this object looks like internally
    scaler.build()

We can do this because 'robust' exists as a default option already
available in BPt (See :ref:`Scalers`). That said, if it wasn't we
could pass it as a custom object as well.

.. ipython:: python

    from sklearn.preprocessing import RobustScaler
    scaler = bp.Scaler(RobustScaler())
    scaler

params
~~~~~~

:ref:`params<params>` are used to either specify or select from a default existing choice
of associated fixed or distribution of hyper-parameter values for this object. For example,
we can choose to associate an existing hyper-parameter distribution for the robust scaler from
before with:

.. ipython:: python

    scaler = bp.Scaler('robust', params="robust gs")
    scaler

We could also set it to a custom distribution using ref:`Parameter<api.dists>`:

.. ipython:: python

    quantile_range = bp.p.Choice([(5, 95), (10, 90), (15, 85)])
    scaler = bp.Scaler('robust', params={'quantile_range': quantile_range})
    scaler

See :ref:`params` for more information on how to set Parameters.


scope
~~~~~~

Pipeline objects also have the argument :ref:`Scope` as an input argument.
This argument allows for pipeline objects to work on just a subset of columns.
Notably, if the scope in reference to a specific :class:`Dataset` is empty, then
the piopeline object in question will just be silently skipped. This is useful for defining
generic pipelines to different types of datasets.

