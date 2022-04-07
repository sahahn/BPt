.. _extra_params:

{{ header }}

**************
Extra Params
**************

All base :class:`ModelPipeline <BPt.ModelPipeline>` have the kwargs style input argument `extra params`. This parameter is designed
to allow passing additional values to the base objects, separate from :ref:`Params`. Take the case where you
are using a preset model, with a preset parameter distribution, but you only want to change 1 parameter in the model while still keeping
the rest of the parameters associated with the param distribution. In this case, you could pass that value in extra params.

`extra params` are passed as in kwargs style, which means as extra named params, where the names
are the names of parameters (only those accessible to the base classes init), for example
if we were selecting the 'dt' ('decision tree') :class:`Model<BPt.Model>`, and we wanted to use the first built in
preset distribution for :ref:`Params`, but then fix the number of `max_features`, we could do it is as:

::

    model = Model(obj = 'dt',
                  params = 1,
                  max_features = 10)

Note: Any parameters passed as extra params will override any values if overlapping with the fixed passed params = 1. In other
words, parameters passed as extra have the highest priority.
