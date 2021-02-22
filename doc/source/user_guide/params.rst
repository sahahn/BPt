.. _params:

{{ header }}

***********
Params
***********

On the back-end, if a :class:`ParamSearch<BPt.ParamSearch>` object is passed when creating a
:class:`ModelPipeline <BPt.ModelPipeline>`, then a hyperparameter search will be conducted.
All Hyperparameter search types are implemented on the backend with facebook's
`Nevergrad <https://github.com/facebookresearch/nevergrad>`_ library, or a sklearn GridSearch.

Specific hyper-parameters distributions in which to search over are set within their corresponding
base ModelPipeline object, e.g., the params argument is :class:`Model<BPt.Model>`. For any object
with a params argument you can set an associated hyperparameter distribution, which specifies values to
search over (again assuming that param_search != None, if param_search is None, only passed params with constant
values will be applied to object of interest, and any with associated Nevergrad parameter distributions will just
be ignored).

You have two different options in terms of input that params can accept, these are:

    - Select a preset distribution
        To select a preset, BPt defined, distribution, the selected object must first
        have at least one preset distribution. These options can be found for each object
        specifically in the documentation under where that object is defined. Specifically,
        they will be listed with both an integer index, and a corresponding str name
        (see :ref:`Models`).
        
        For example, in creating a binary :class:`Model<BPt.Model>` we could pass:
        
        ::
            
            # Option 1 - as int
            model = Model(obj = "dt classifier",
                          params = 1)

            # Option 2 - as str
            model = Model(obj = "dt classifier",
                          params = "dt classifier dist")

        In both cases, this selects the same preset distribution for the decision
        tree classifier.


    - Pass a custom nevergrad distribution
        If you would like to specify your own custom hyperparameter distribution to search over,
        you can, you just need to specify it as a python dictionary of 
        `nevergrad parameters <https://facebookresearch.github.io/nevergrad/parametrization.html>`_ 
        (follow the link to learn more about how to specify nevergrad params).
        You can also go into the source code for BPt, specifically BPt/helpers/Default_Params.py,
        to see how the preset distributions are defined, as a further example.

        Specifically the dictionary of params should follow the scikit_learn param dictionary format,
        where the each key corresponds to a parameter, but the value as a nevergrad parameter (instead of scikit_learn style).
        Further, if you need to specify nested parameters, e.g., for a custom object, you separate parameters with '__',
        so e.g., if your custom model has a base_estimator param, you can pass:
        
        ::

            params = {'base_estimator__some_param' : nevergrad dist}

        Lastly, it is worth noting that you can pass either just static values or a combination of nevergrad distributions
        and static values, e.g.,

        ::

            {'base_estimator__some_param' : 6} 

        (Note: extra params can also be used to pass static values, and extra_params takes precedence
        if a param is passed to both params and extra_params).

The special input wrapper :class:`Select<BPt.Select>`
can also be used to implicitly introduce hyper-parameters
into the :class:`ModelPipeline <BPt.ModelPipeline>`. 
