from typing import Tuple, Dict, Iterable

import torch
import torch.distributed as dist
from torch import Tensor


def reduce_tensor(tensor, mean: bool = True) -> torch.Tensor:
    with torch.no_grad():
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        if mean:
            rt *= (1. / dist.get_world_size())
        return rt


def reduce_loss_watch(loss: torch.Tensor,
                      watch: Dict[str, torch.Tensor]
                      ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    keys = sorted(list(watch.keys()))
    all = torch.stack([loss.detach()] + [watch[k].detach() for k in keys])
    all = reduce_tensor(all, mean=True)
    # as losses are already averaged on each worker
    new_watch = {k: all[i + 1] for i, k in enumerate(keys)}
    return all[0], new_watch


def get_pg_for_syncbn(world_size: int, rank: int, local_rank: int):

    all_local_rank = torch.empty(world_size, dtype=torch.long, device='cuda')
    cur_local_rank = torch.empty((), dtype=torch.long, device='cuda')\
        .fill_(local_rank)

    dist.all_gather(list(all_local_rank.unbind(0)), cur_local_rank,
                    group=dist.group.WORLD,
                    async_op=False)

    all_rank       = torch.arange(end=world_size, dtype=torch.long, device='cuda')
    pg_ids         = (all_rank - all_local_rank).cpu().tolist()

    same_node      = [i for i in range(world_size)
                      if pg_ids[i] == pg_ids[rank]]
    uniq_pg_ids    = set(pg_ids)

    if len(same_node) == world_size:
        print('single node, use global pg')
        return None
    else:
        print('multi node, my rank is {}, ranks in the same node {}'
              .format(rank, same_node))
        pgs = {}
        ret = None
        for cur_pg in uniq_pg_ids:
            pg_ranks = [i for i in range(world_size)
                        if pg_ids[i] == cur_pg]
            pgs[cur_pg] = dist.new_group(pg_ranks)
            if set(pg_ranks) == set(same_node):
                ret = pgs[cur_pg]
        print(pgs)
        assert ret is not None
        return ret


def cuda_preload_count(aggr: Iterable[Dict[str, Tensor]],
                       count: int = None
                       ) -> Iterable[Dict[str, Tensor]]:

    if not torch.cuda.is_available():
        if count is None:
            return aggr
        else:
            iter_aggr = iter(aggr)
            for cnt in range(count):
                try:
                    yield next(iter_aggr)
                except StopIteration:
                    iter_aggr = iter(aggr)
                    yield next(iter_aggr)

    stream = torch.cuda.Stream()

    loading_batch: Dict[str, Tensor] = None
    ready_batch: Dict[str, Tensor]   = None

    actual_count = count if count is not None else len(aggr)
    iter_aggr = iter(aggr)
    for cnt in range(actual_count):
        try:
            batch = next(iter_aggr)
        except StopIteration:
            iter_aggr = iter(aggr)
            batch = next(iter_aggr)

        assert isinstance(batch, dict)

        # synchronize so the loading is complete,
        # and then move them to "ready_*"
        torch.cuda.current_stream().wait_stream(stream)
        ready_batch = loading_batch

        # start loading the next batch
        with torch.cuda.stream(stream):
            loading_batch = {k: v.cuda(non_blocking=True)
                             if isinstance(v, Tensor) else v
                             for k, v in batch.items()}

        if ready_batch is not None:
            for v in ready_batch.values():
                if isinstance(v, Tensor):
                    v.record_stream(torch.cuda.current_stream())
            yield ready_batch

    torch.cuda.current_stream().wait_stream(stream)
    ready_batch = loading_batch

    if ready_batch is not None:
        for v in ready_batch.values():
            if isinstance(v, Tensor):
                v.record_stream(torch.cuda.current_stream())
        yield ready_batch
