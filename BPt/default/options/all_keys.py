def get_all_keys():

    # Start with a few used for testing errors and whatnot
    all_keys = set(['fake', 'default'])

    from .ensembles import all_obj_keys
    all_keys.update(all_obj_keys)

    from .feature_selectors import all_obj_keys
    all_keys.update(all_obj_keys)

    from .imputers import all_obj_keys
    all_keys.update(all_obj_keys)

    from .loaders import all_obj_keys
    all_keys.update(all_obj_keys)

    from .models import all_obj_keys
    all_keys.update(all_obj_keys)

    from .scalers import all_obj_keys
    all_keys.update(all_obj_keys)

    from .transformers import all_obj_keys
    all_keys.update(all_obj_keys)

    from ...extensions.mv_transformer.options import all_obj_keys
    all_keys.update(all_obj_keys)

    return all_keys
