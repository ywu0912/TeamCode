import os.path as osp
import importlib
import inspect
import glob


def create_model(cfg, log):

    filenames = glob.glob(osp.join(osp.dirname(__file__), '*', 'model.py'))
    filenames = [fn.split(osp.sep)[-2] for fn in filenames]
    type2model = dict()
    for filename in filenames:
        module = importlib.import_module('models.%s.model' % filename)
        clsmembers = inspect.getmembers(module, inspect.isclass)
        for clsmember in clsmembers:
            if clsmember[0].startswith(filename):
                type2model[clsmember[0]] = clsmember[1]
    print('type2model=>', type2model)
    return type2model[cfg.model_type](cfg, log)
