# Copyright 2018-YuejiaXiang, NLP Lab., Northeastern university
# This code is used to monitor gpu memory usage.
#
# How to use:
# from trackMemory import track
# gpu_id = 3
# myTrack = track(gpu_id)
# myTrack.showMemory("line 31")
#
import pynvml


class Track:
    def __init__(self, gpu_index, scale='Auto', ignore_zero=False):
        pynvml.nvmlInit()
        if scale == 'Auto':
            pass
        elif scale == 'K':
            self.scale = 1024
        elif scale == 'M':
            self.scale = 1024**2
        elif scale == 'G':
            self.scale = 1024**3
        else:
            print 'unknown scale name: %s' % scale
        self.scaleName = scale
        self.ignoreZero = ignore_zero
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self.mem_used = pynvml.nvmlDeviceGetMemoryInfo(self.handle).used
        self.mem_init = self.mem_used

    def show_memory_add(self, line):
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle).used
        mem_add = mem_info - self.mem_used
        self.mem_used = mem_info
        if not self.ignoreZero or mem_add != 0:
            if self.scaleName == 'Auto':
                if mem_add >= 1024**3:
                    print('[%s] ' % line + "gpu cost: " + str(mem_add / 1024**3) + 'G')
                elif mem_add >= 1024**2:
                    print('[%s] ' % line + "gpu cost: " + str(mem_add / 1024**2) + 'M')
                elif mem_add >= 1024**1:
                    print('[%s] ' % line + "gpu cost: " + str(mem_add / 1024**1) + 'K')
                else:
                    print('[%s] ' % line + "gpu cost: " + str(mem_add) + 'B')
            else:
                print('[%s] ' % line + "gpu cost: " + str(mem_add/self.scale) + self.scaleName)