from copy import deepcopy
import torch
from torch.utils.data import Dataset, ConcatDataset
from os import path
from typing import Tuple, List, Union, TextIO, NamedTuple, Dict
import numpy as np
from torch import Tensor
import cv2
from imgaug.parameters import Binomial
from transform import Transform


class PtsFileDesc:
    """
    version: 1
    n_points: n
    {
    x y
    x y
    ...
    x y
    }

    """

    def __init__(self, *, filepath: str = None, array: np.ndarray = None) -> None:
        assert (filepath is None) ^ (array is None)
        if filepath is not None:
            with open(filepath, 'r') as f:
                lines = [line.strip() for line in f.readlines()]
            assert lines[0] == 'version: 1'
            assert lines[1].startswith('n_points: ')
            numpt = int(lines[1][len('n_points: '):])
            assert lines[2] == '{'
            lines = lines[3:]
            assert lines[numpt] == '}'

            array = np.empty((numpt, 2), dtype=np.float32)
            for i in range(numpt):
                parts = lines[i].split(' ')
                assert len(parts) == 2
                array[i, 0] = float(parts[0])
                array[i, 1] = float(parts[1])

        else:
            assert array.ndim == 2 and array.shape[1] == 2
            array = array.astype(np.float32)

        self.points = array

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    def __str__(self) -> str:
        return \
            f'version: 1\n' + \
            f'n_points: {self.n_points}\n' + \
            f'{{\n' + \
            '\n'.join([f'{self.points[i, 0]} {self.points[i, 1]}'
                       for i in range(self.n_points)]) + '\n' + \
            f'}}\n'

    def save_to_file(self, file: Union[str, TextIO]) -> None:
        if isinstance(file, str):
            with open(file, 'w') as f:
                f.write(str(self))
        else:
            file.write(str(self))


class Sample(NamedTuple):
    image: str
    pts: PtsFileDesc


class PtsDataset(Dataset):
    """
    listfile should be like formet below:

    roordir/a/a1.jpg rootdir/a/a1.pts
    rootdir/a/a2.jpg rootdir/a/a2.pts
    ...

    where rootdir is the directory where listfile is placed in
    """
    def __init__(self, listfile: str, numpt: int, transform: Transform,
                 mirror_mapping: List[Tuple[int, int]] = [],
                 training: bool = False):
        super().__init__()
        assert numpt > 0
        if True:
            for t in mirror_mapping:
                assert len(t) == 2
            test_set = set()
            for t in mirror_mapping:
                test_set = test_set.union(t)
            assert len(test_set) == 2 * len(mirror_mapping)

        self.training = bool(training)
        self.numpt = numpt
        self.mirror_mapping = mirror_mapping
        self.rootdir = path.dirname(listfile)
        with open(listfile, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
            for line in lines:
                assert len(line.split(' ')) == 2
            lines = [tuple(line.split(' ')) for line in lines]
        self.samples = [Sample(t[0], PtsFileDesc(filepath=self.rootdir + '/' + t[1]))
                        for t in lines]
        self.fliplr = Binomial(0.5)

        self.transform = transform
        mirror_index = np.array(self.mirror_mapping, dtype=np.int64)
        self.mirror_index = (np.concatenate((mirror_index[:, 0], mirror_index[:, 1]), axis=0),
                             np.concatenate((mirror_index[:, 1], mirror_index[:, 0]), axis=0))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(f'Out of Range [0, {len(self - 1)}]')

        sample = self.samples[index]
        image = cv2.imread(self.rootdir + '/' + sample.image, cv2.IMREAD_COLOR)
        pts = deepcopy(sample.pts.points)

        if self.training and self.fliplr.draw_sample() == 1:
            image, pts = self._fliplr(image, pts)

        return self.transform({'image': torch.from_numpy(image), 'pts': torch.from_numpy(pts)})

    def _fliplr(self, image: np.ndarray, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        image = np.ascontiguousarray(image[:, ::-1, :])
        pts[:, 0] = image.shape[1] - pts[:, 0]
        pts[self.mirror_index[0], :] = pts[self.mirror_index[1], :]
        return image, pts


def build_dataset(training: bool) -> ConcatDataset:
    from config import cfg
    train_trans = Transform(
        cfg.MODEL.INPUT_SIZE,
        rotate=cfg.AUGMENTATION.ROTATE,
        rotate_step=cfg.AUGMENTATION.ROTATE_STEP,
        scale=tuple(cfg.AUGMENTATION.SCALE),
        translate_ratio=cfg.AUGMENTATION.TRANSLATE_RATIO,
        bbox_ratio=cfg.AUGMENTATION.BBOX_RATIO,
    )
    eval_trans = Transform(
        cfg.MODEL.INPUT_SIZE,
        bbox_ratio=cfg.AUGMENTATION.BBOX_RATIO,
    )
    datasets = [PtsDataset(
        listfile,
        cfg.DEFINITION.NUM_PT,
        deepcopy(train_trans) if training else deepcopy(eval_trans),
        cfg.DEFINITION.MIRROR_MAPPING,
        training=training,
    ) for listfile in (cfg.DATASETS if training else cfg.VAL_DATASETS)]

    return ConcatDataset(datasets)


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Inputs:
        =num_cls=
            Number of classes.
    Returns:
        The color map.
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


PALETTE = get_palette(257)


def visualize(image: Tensor, pts: Tensor) -> np.ndarray:

    from transform import get_inverse_transform
    import math

    inv_trans = get_inverse_transform()
    npimage = inv_trans(image.cpu())
    nppts = pts.cpu().reshape(-1, 2).numpy()

    thick = int(math.ceil(max(npimage.shape) / 200.0))
    for i in range(nppts.shape[0]):
        x, y = nppts[i, :]
        print(x, y)
        cv2.circle(npimage, (int(x * 16), int(y * 16)), 16,
                   color=PALETTE[i * 3 + 3: i * 3 + 6],
                   thickness=thick, shift=4)
    return npimage


if __name__ == '__main__':

    import sys
    from config import merge_from_file

    merge_from_file(sys.argv[1])
    ds = build_dataset(True)

    for i in range(10):
        s = ds[0]
        assert 'image' in s and 'pts' in s
        show = visualize(s['image'], s['pts'])
        cv2.imwrite(f'trans_{i}.png', show[:, :, ::-1])
