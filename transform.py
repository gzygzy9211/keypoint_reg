from typing import Callable, Dict, Optional, Tuple, List
from imgaug import parameters
import numpy as np
import numba
from torch import Tensor, from_numpy
from torchvision.transforms import Compose, Normalize, ToTensor
import cv2
from PIL.Image import Image

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class BGR2RGB:
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(img[:, :, ::-1])


class InverseNormalize:

    def __init__(self, mean: List[float], std: List[float]):
        assert len(mean) == len(std)
        assert all((v > 0 for v in std))
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, t: Tensor) -> Tensor:
        return t.mul(from_numpy(self.std).view(-1, 1, 1))\
            .add_(from_numpy(self.mean).view(-1, 1, 1))


class PILToNumpy:

    def __call__(self, pil: Image) -> np.ndarray:
        return np.array(pil)


class Transform:

    def __init__(self, output_size: int, *,
                 rotate: float = 0.,
                 rotate_step: Optional[float] = None,
                 scale: Tuple[float, float] = (1., 1.),
                 translate_ratio: float = 0.,
                 bbox_ratio: Optional[float] = None) -> None:
        # bbox_ratio: move the target to the center according to bbox of keypoints,
        #             and then scale to the size that bbox_size * bbox_ratio = image_size,
        #             and finally apply random affine
        assert bbox_ratio is None or (bbox_ratio > 0.5 and bbox_ratio < 20.)
        self.bbox_ratio = bbox_ratio

        assert rotate >= 0.
        self.rotate = rotate

        assert rotate_step is None or (rotate_step > 0. and rotate_step <= 180.)
        self.rotate_step = rotate_step

        scale = (min(scale), max(scale))
        assert scale[0] > 0.
        self.scale = scale

        assert translate_ratio >= 0. and translate_ratio <= 0.5
        self.translate_ratio = translate_ratio

        assert output_size > 0
        self.output_size = output_size

        self.rotate_sampler = parameters.RandomSign(parameters.Choice(
            [self.rotate_step * i for i in
             range(int(self.rotate / self.rotate_step))], True
        )) if self.rotate_step is not None else \
            parameters.Uniform(- self.rotate, self.rotate)
        self.scale_sampler = parameters.Uniform(*self.scale)
        self.translate_sampler = parameters.Uniform(-self.translate_ratio, self.translate_ratio)

        # TODO: maybe add some noise from iaa?
        self.image_transform = Compose([
            BGR2RGB(),
            ToTensor(),
            Normalize(mean=MEAN, std=STD),
        ])

    def __call__(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        image = sample['image'].numpy()
        pts = sample['pts'].numpy()
        trans = self.get_affine_matrix(image, pts)
        image, pts = self.apply_affine_matrix(image, pts, trans)
        return {'image': self.image_transform(image),
                'pts': from_numpy(pts.reshape(-1)),
                'inverse_trans': from_numpy(np.linalg.inv(trans))}

    BORDER_VALUE = np.array([0.485, 0.456, 0.406][::-1]) * 255

    def apply_affine_matrix(self, image: np.ndarray, pts: np.ndarray, trans: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        image = cv2.warpAffine(image, trans[0:2], (self.output_size, self.output_size),
                               borderMode=cv2.BORDER_CONSTANT, borderValue=self.BORDER_VALUE)
        pts = (np.matmul(trans[0:2, 0:2], pts.T) + trans[0:2, 2:3]).T
        return image, pts

    def get_affine_matrix(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        assert pts.ndim == 2 and pts.shape[1] == 2

        scale = self.scale_sampler.draw_sample()
        rotate = self.rotate_sampler.draw_sample()
        translate = self.translate_sampler.draw_samples(2)

        ocx = ocy = (self.output_size - 1) * 0.5  # warpAffine use align_corners=True

        preproc = self._centerize(pts, self.output_size, self.bbox_ratio) if self.bbox_ratio is not None \
            else self._centerize(np.array([[0, 0], image.shape[0:2][::-1]], dtype=np.float32),
                                 self.output_size, 1.,)

        t_scale = self._scale(scale, ocx, ocy)

        t_rotate = self._rotate(rotate, ocx, ocy)

        t_translate = self._translate(translate[0], translate[1], self.output_size)

        return np.einsum('ij,jk,kl,lm->im', t_translate, t_rotate, t_scale, preproc)

    @staticmethod
    @numba.njit
    def _centerize(pts: np.ndarray, output_size: int, bbox_ratio: float) -> np.ndarray:
        trans = np.eye(3, dtype=np.float32)
        left = np.min(pts[:, 0])
        righ = np.max(pts[:, 0])
        top = np.min(pts[:, 1])
        bot = np.max(pts[:, 1])
        cx = 0.5 * (left + righ)
        cy = 0.5 * (top + bot)
        size = max(righ - left, bot - top) * bbox_ratio

        halfsize = size * 0.5
        left = cx - halfsize
        righ = cx + halfsize
        top = cy - halfsize
        bot = cy + halfsize

        scale = output_size / size
        trans[0, 0] = scale
        trans[1, 1] = scale
        trans[0, 2] = - left * scale
        trans[1, 2] = - top * scale
        return trans

    @staticmethod
    @numba.njit
    def _scale(scale: float, cx: float, cy: float) -> np.ndarray:
        trans = np.eye(3, dtype=np.float32)
        trans[0, 0] = scale
        trans[1, 1] = scale
        trans[0, 2] = (1. - scale) * cx
        trans[1, 2] = (1. - scale) * cy
        return trans

    @staticmethod
    @numba.njit
    def _rotate(angle: float, cx: float, cy: float) -> np.ndarray:
        # positive -> clock-wise
        # dst_x - cx = cos * (src_x - cx) - sin * (src_y - cy)
        # dst_y - cy = sin * (src_x - cx) + cos * (src_y - cy)
        rad = angle * np.pi / 180.
        cos = np.cos(rad)
        sin = np.sin(rad)
        trans = np.eye(3, dtype=np.float32)
        trans[0, 0] = cos
        trans[0, 1] = -sin
        trans[1, 0] = sin
        trans[1, 1] = cos
        trans[0, 2] = cx - cos * cx + sin * cy
        trans[1, 2] = cy - sin * cx - cos * cy
        return trans

    @staticmethod
    @numba.njit
    def _translate(ratio_x: float, ratio_y: float, size: int) -> np.ndarray:
        trans = np.eye(3, dtype=np.float32)
        trans[0, 2] = ratio_x * size
        trans[1, 2] = ratio_y * size
        return trans


def get_inverse_transform() -> Callable[[Tensor], np.ndarray]:
    from torchvision.transforms import ToPILImage
    return Compose([
        InverseNormalize(mean=MEAN, std=STD),
        ToPILImage(),
        PILToNumpy(),
        BGR2RGB(),
    ])


if __name__ == '__main__':
    pass
