import time
import torch
from pathlib import Path


class TimeTable(object):
    def __init__(self):
        self.csv = None
        self.reset_epoch()

    def reset_epoch(self):
        self.t_epoch = 0.0
        self.t_loop = 0.0
        self.t_eval = 0.0
        self.t_forward = 0.0
        self.t_backward = 0.0
        self.t_sample = 0.0
        self.t_prep_batch = 0.0
        self.t_prep_input = 0.0
        self.t_post_update = 0.0
        self.t_time_zero = 0.0
        self.t_time_nbrs = 0.0
        self.t_attn_split = 0.0
        self.t_attn_cat = 0.0
        self.t_attn_wmul = 0.0
        self.t_attn_repeat = 0.0
        self.t_attn_reshape = 0.0
        self.t_attn_qkdot = 0.0
        self.t_attn_act = 0.0
        self.t_attn_softmax = 0.0
        self.t_attn_dropout = 0.0
        self.t_attn_vmul = 0.0
        self.t_attn_vcat = 0.0
        self.t_attn_vsum = 0.0
        self.t_attn_ffn = 0.0

    def start(self):
        torch.cuda.synchronize()
        return time.perf_counter()

    def elapsed(self, start):
        torch.cuda.synchronize()
        return time.perf_counter() - start

    def print_epoch(self, prefix='  '):
        lines = f'' \
            f'{prefix}epoch | total:{self.t_epoch:.2f}s loop:{self.t_loop:.2f}s eval:{self.t_eval:.2f}s\n' \
            f'{prefix} loop | forward:{self.t_forward:.2f}s backward:{self.t_backward:.2f}s sample:{self.t_sample:.2f}s prep_batch:{self.t_prep_batch:.2f}s prep_input:{self.t_prep_input:.2f}s post_update:{self.t_post_update:.2f}s\n' \
            f'{prefix} time | zero:{self.t_time_zero:.2f}s nbrs:{self.t_time_nbrs:.2f}s\n' \
            f'{prefix} attn | split:{self.t_attn_split:.2f}s cat:{self.t_attn_cat:.2f}s wmul:{self.t_attn_wmul:.2f}s repeat:{self.t_attn_repeat:.2f}s reshape:{self.t_attn_reshape:.2f}s qkdot:{self.t_attn_qkdot:.2f}s act:{self.t_attn_act:.2f}s softmax:{self.t_attn_softmax:.2f}s dropout:{self.t_attn_dropout:.2f}s vmul:{self.t_attn_vmul:.2f}s vcat:{self.t_attn_vcat:.2f}s vsum:{self.t_attn_vsum:.2f}s\n'
        print(lines, end='')

    def csv_open(self, path):
        self.csv_close()
        self.csv = Path(path).open('w')

    def csv_close(self):
        if self.csv is not None:
            self.csv.close()
            self.csv = None

    def csv_write_header(self):
        header = 'epoch,total,loop,eval,' \
            'forward,backward,sample,prep_batch,prep_input,post_update,' \
            'time_zero,time_nbrs,' \
            'split,cat,wmul,repeat,reshape,qkdot,act,softmax,dropout,vmul,vcat,vsum'
        self.csv.write(header + '\n')

    def csv_write_line(self, epoch):
        line = f'{epoch},{self.t_epoch},{self.t_loop},{self.t_eval},' \
            f'{self.t_forward},{self.t_backward},{self.t_sample},{self.t_prep_batch},{self.t_prep_input},{self.t_post_update},' \
            f'{self.t_time_zero},{self.t_time_nbrs},' \
            f'{self.t_attn_split},{self.t_attn_cat},{self.t_attn_wmul},{self.t_attn_repeat},{self.t_attn_reshape},{self.t_attn_qkdot},{self.t_attn_act},{self.t_attn_softmax},{self.t_attn_dropout},{self.t_attn_vmul},{self.t_attn_vcat},{self.t_attn_vsum}'
        self.csv.write(line + '\n')


# Global for accumulating timings.
tt = TimeTable()
