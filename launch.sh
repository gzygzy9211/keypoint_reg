# $/bin/bash
# patch pytorch 1.6.0 SyncBN implementation to prevent error on local "process group"
sed -i 's$dist.all_gather(combined_list, combined, async_op=False)$dist.all_gather(combined_list, combined, process_group, async_op=False)$g' \
    `python3 -c "import torch; print(torch.nn.modules._functions.__file__ if torch.__version__.startswith('1.6.0') else '')"` || \
    echo 'no need SyncBN patch'

python3 -m pip install -i PyYAML==5.3 imgaug==0.4 easydict==1.9 numba==0.53.1

SRC_DIR=$(dirname "$0")
NUM_GPU=2
PORT=`python3 -c "import random; print(20000 + random.randint(0, 20000))"`

python3 $SRC_DIR/main.py --help

set -x
python3 -u -m torch.distributed.launch --nproc_per_node=$NUM_GPU  --master_port $PORT \
    $SRC_DIR/main.py \
    $@
