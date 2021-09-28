from copy import deepcopy
from easydict import EasyDict as EZ

__C = EZ()
cfg = __C

# defaults goes here

cfg.DEFINITION = EZ()
cfg.DEFINITION.NUM_PT = 8
cfg.DEFINITION.MIRROR_MAPPING = []

cfg.DATASETS = []
cfg.VAL_DATASETS = []

cfg.AUGMENTATION = EZ()
cfg.AUGMENTATION.ROTATE = 30
cfg.AUGMENTATION.ROTATE_STEP = None
cfg.AUGMENTATION.SCALE = [1, 1]
cfg.AUGMENTATION.TRANSLATE_RATIO = 0.25
cfg.AUGMENTATION.BBOX_RATIO = 2

cfg.MODEL = EZ()
cfg.MODEL.BACKBONE = 'resnet18'
cfg.MODEL.INPUT_SIZE = 224
cfg.MODEL.HIDDEN_DIM = 256
cfg.MODEL.NUM_HIDDEN = 1
cfg.MODEL.HIDDEN_ACT = 'relu'
cfg.MODEL.HIDDEN_ACT_KWARGS = {}

cfg.TRAIN = EZ()
cfg.TRAIN.BATCH_SIZE = 128
cfg.TRAIN.OPTIMIZER = 'sgd'
cfg.TRAIN.LR = 0.01
cfg.TRAIN.WD = 5e-4
cfg.TRAIN.EPOCH = 120
cfg.TRAIN.LOSS = 'smoothl1'

__defualt = deepcopy(cfg)


def reset_to_default():
    global cfg
    __C = deepcopy(__defualt)
    cfg = __C


def merge_from_file(path: str):
    import yaml
    import yaml.representer
    import io

    with open(path, 'r') as f:
        new_cfg = yaml.load(f, Loader=yaml.SafeLoader)
    new_cfg = EZ(new_cfg)

    def represent_none(self, _):
        return self.represent_scalar('tag:yaml.org,2002:null', '')

    def updater(origin: EZ, new: EZ):
        for key, val in origin.items():
            if key in new:
                assert not (
                    isinstance(origin[key], EZ) ^ isinstance(new[key], EZ)
                )
                if isinstance(origin[key], EZ):
                    updater(origin[key], new[key])
                else:
                    origin[key] = new[key]

    updater(cfg, new_cfg)
    yaml.SafeDumper.add_representer(type(None), represent_none)
    yaml.SafeDumper.add_representer(EZ, yaml.representer.SafeRepresenter.represent_dict)

    with io.StringIO() as stream:
        yaml.dump(cfg, stream, Dumper=yaml.SafeDumper)
        stream.seek(io.SEEK_SET, 0)
        print(f'Current Config is:\n{stream.read()}')


if __name__ == '__main__':
    import sys
    merge_from_file(sys.argv[1])
