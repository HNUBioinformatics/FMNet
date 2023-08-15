import subprocess
import h5py
import glob
import os


class RunCmd(object):
    def cmd_run(self, cmd):
        self.cmd = cmd
        subprocess.call(self.cmd, shell=True)


# Train and test the model
for i in range(2):
        print('train：',i)
        a = RunCmd()
        a.cmd_run('python FMNet.py \
            --posi=./GraphProt_CLIP_sequences/ZC3H7B_Baltz2012.train.positives.fa \
            --nega=./GraphProt_CLIP_sequences/ZC3H7B_Baltz2012.train.negatives.fa\
            --train=True --n_epochs=50')
    for j in range(10):
      a = RunCmd()
      a.cmd_run('python FMNet.py \
              --testfile=./GraphProt_CLIP_sequences/ZC3H7B_Baltz2012.ls.positives.fa \
              --nega=./GraphProt_CLIP_sequences/ZC3H7B_Baltz2012.ls.negatives.fa\
              --predict=True')





