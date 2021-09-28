from config import merge_from_file
import torch
import torch.multiprocessing
import torch.distributed as dist
from runtime import Runtime
from training import KeyPointTraining

if __name__ == '__main__':
    from argparse import ArgumentParser
    import time

    parser = ArgumentParser()
    parser.add_argument('config_file', type=str)
    parser.add_argument('--checkpoint_path', type=str, default='./ckpt')
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(args.local_rank)

    dist.init_process_group(backend='nccl', init_method='env://')

    Runtime.world_size = dist.get_world_size()
    Runtime.lr_scale = float(dist.get_world_size())
    Runtime.rank = dist.get_rank()
    Runtime.local_rank = args.local_rank
    Runtime.print_freq = args.print_freq
    Runtime.num_workers = args.num_workers

    merge_from_file(args.config_file)

    training = KeyPointTraining(args.checkpoint_path, args.pretrain, True)
    while not training.should_stop:
        start = time.time()
        training.train_epoch()
        end = time.time()
        if Runtime.local_rank == 0:
            print(f'Epoch {training.epoch} Duration: {end - start:.3f} secs.')
