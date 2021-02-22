.. _pipeline_objects:

{{ header }}

*******************
Pipeline Objects
*******************

.. currentmodule:: BPt

The pipeline classes :class:`Pipeline` and :class:`ModelPipeline`
are made up of base pipeline objects / pieces.

Across all base :class:`ModelPipeline<BPt.ModelPipeline>` pieces,
e.g., :class:`Model<BPt.Model>` or :class:`Scaler<BPt.Scaler>`,
there exists an `obj` param when initializing these objects. This parameter
can broadly refer to either a str, 
which indicates a valid pre-defined custom obj for that piece, or depending
on the pieces, this parameter can be passed a custom object directly.