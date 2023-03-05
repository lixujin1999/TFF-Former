# TFF-Former

# Linux operation instruction:
python -m torch.distributed.launch --master_port 29502 --nproc_per_node=4 TFF-Former-RSVP/run.py

python -m torch.distributed.launch --master_port 29502 --nproc_per_node=4 TFF-Former-SSVEP/run.py

# benchmark dataset:
Wang, Y., Chen, X., Gao, X., & Gao, S. (2016). A benchmark dataset for SSVEP-based brain–computer interfaces. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 25(10), 1746-1752.
