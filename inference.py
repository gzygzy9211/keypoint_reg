# this file demostrate how to inference a model
# also provide command line interface to inference a single picture
import numpy as np
from transform import Transform
from typing import Dict, Optional
from torch import nn
from training import KeyPointTraining
from config import merge_from_file, reset_to_default
import torch
from model import KeyPointModel
import cv2
from dataset import visualize


# in a NOT-Training script, use this function to construct
#     KeyPointModel instance for inference ONLY
def get_inference_model(config: str, ckpt: str) -> KeyPointModel:
    reset_to_default()
    merge_from_file(config)
    training = KeyPointTraining(ckpt, None, False)
    assert training.epoch > 0
    m = training.model
    if isinstance(m, nn.parallel.DistributedDataParallel):
        m = m.module
    assert isinstance(m, KeyPointModel)
    m.cpu().eval()
    return m


# given KeyPointModel instance and external bbox_ratio config, return
#     a Transform instance which can provide preprocessing functionality
#     corresponding to the given model and bbox prior situation
def get_transform(model: KeyPointModel, bbox_ratio: Optional[float]) -> Transform:
    return Transform(model.input_size, bbox_ratio=bbox_ratio)


# construct an input sample from given image and optional one of bbox in
#     left-top-right-bottom representation or points from pts file, which
#     can be fed into the Transform instance to produce input for the model
def build_input(image: np.ndarray, *,
                ltrb_str: Optional[str] = None,
                pts_file: Optional[str] = None) -> Dict[str, torch.Tensor]:
    from dataset import PtsFileDesc
    if pts_file is not None:
        pts_desc = PtsFileDesc(filepath=pts_file)
        pts = pts_desc.points
    elif ltrb_str:
        left, top, right, bottom = [float(v) for v in ltrb_str.split(',')]
        assert right > left
        assert bottom > top
        pts = np.array([[left, top], [right, bottom]], dtype=np.float32)
    else:
        pts = np.array([[0, 0], image.shape[0:2]], dtype=np.float32)

    return {'image': torch.from_numpy(image), 'pts': torch.from_numpy(pts)}


def final_visualize(npimage: np.ndarray, pts: torch.Tensor) -> np.ndarray:

    import math
    from dataset import PALETTE

    nppts = pts.cpu().reshape(-1, 2).numpy()

    thick = int(math.ceil(max(npimage.shape) / 200.0))
    for i in range(nppts.shape[0]):
        x, y = nppts[i, :]
        print(x, y)
        cv2.circle(npimage, (int(x * 16), int(y * 16)), 16,
                   color=PALETTE[i * 3 + 3: i * 3 + 6][::-1],
                   thickness=thick, shift=4)
    return npimage


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('image', type=str)
    parser.add_argument('config_file', type=str)
    parser.add_argument('--checkpoint_path', type=str, default='./ckpt')
    parser.add_argument('--bbox_ratio', type=float, default=None,
                        help='externally specified bbox_ratio indicating given image '
                        'has a bbox prior (positive float for with prior, None for w/o)')
    parser.add_argument('--bbox', type=str, default=None,
                        help='used when image has a bbox prior (bbox_ratio is not None), '
                        'represented in left,top,right,bottom format')
    parser.add_argument('--pts', type=str, default=None,
                        help='used when image has a bbox prior (bbox_ratio is not None), '
                        'a set of points in pts file whose bounding box is the prior')

    args = parser.parse_args()

    torch.cuda.set_device(0)

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    assert image is not None

    with torch.no_grad():
        # init step 1. construct the inference model instance
        model = get_inference_model(args.config_file, args.checkpoint_path)
        # init step 2. construct the transform instance for preprocessing
        transform = get_transform(model, args.bbox_ratio)

        # run step 1. construct input sample (a dict)
        sample = build_input(image, ltrb_str=args.bbox, pts_file=args.pts)
        # run step 2. perform preprocessing via the transform instance
        sample = transform(sample)
        # run step 3. model forward
        pred_pts = model(sample['image'].unsqueeze(0)).reshape(-1, 2)
        # run step 4. transform the predicted pts back into coordinate of origin image
        #     using 'inverse_trans' provided by transform
        inv_trans = sample['inverse_trans']
        final_pred_pts = (torch.matmul(inv_trans[0:2, 0:2], pred_pts.t()) +
                          inv_trans[0:2, 2:3]).t()

    # visualization
    predict_image = visualize(sample['image'], pred_pts)
    final_image = final_visualize(image, final_pred_pts)

    cv2.imwrite('network_output.png', predict_image)
    cv2.imwrite('final_output.png', final_image)
