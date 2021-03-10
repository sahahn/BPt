.. _saving_data:

{{ header }}

**************
Saving Data
**************

.. currentmodule:: BPt

The best way to save :class:`Dataset` is through the pandas base method :func:`pandas.DataFrame.to_pickle`.
Likewise, you can load saved :class:`Dataset` with the pandas function :func:`pandas.read_pickle`. These methods
are also available through the BPt namespace.

Below we create and save a dataset

.. ipython:: python

    import BPt as bp
    
    data = bp.Dataset()
    data['a1'] = [1, 2, 3]
    data['a2'] = [1, 2, 3]
    data['t'] = [1, 1, 1]
    
    data = data.add_scope('a', 'custom_scope')
    data['custom_scope']

    data.to_pickle('my_data.pkl')
    del data

Now we can try loading it:

.. ipython:: python
    
    data = bp.read_pickle('my_data.pkl')
    data['custom_scope']
    
And delete it when we are done

.. ipython:: python

    import os
    os.remove('my_data.pkl')


    
