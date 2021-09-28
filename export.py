from model import KeyPointModel
from config import merge_from_file
import torch
from training import KeyPointTraining
import torch.onnx

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('config_file', type=str)
    parser.add_argument('--checkpoint_path', type=str, default='./ckpt')

    args = parser.parse_args()

    torch.cuda.set_device(0)

    merge_from_file(args.config_file)

    training = KeyPointTraining(args.checkpoint_path, None, False)
    assert training.epoch > 0
    m = training.model
    if isinstance(m, torch.nn.parallel.DistributedDataParallel):
        m = m.module
    assert isinstance(m, KeyPointModel)
    device = next(m.parameters()).device
    assert len(training.val_dataset) > 0
    sample = training.val_dataset[0]
    output = m(sample['image'].unsqueeze(0).to(device))

    print(output)
    torch.save({'input': sample['image'].unsqueeze(0), 'output': output},
               args.checkpoint_path + f'/epoch_{training.epoch}.onnx.io')

    torch.onnx.export(m, (sample['image'].unsqueeze(0).to(device),),
                      args.checkpoint_path + f'/epoch_{training.epoch}.onnx',
                      opset_version=11)
