# FMNet: Multi-network fusion model for RNA-protein binding sites prediction

****
**Introduction**

  In this study, we developed FMNet to predict RBPs binding sites, aiming at the problem that existing models can’t obtain sufficient information from RNA sequences. FMNet uses weighted average method to fuse the RSNet, ReNet, CNNs, SACBiLSTM and SACBiGRU. More sequence feature information is extracted by FMNet through multi-scale windows. The FMNet can identify RBPs binding sites with high accuracy using only sequence information without using additional features, and uses CNN to identify binding motifs in RNA sequences, which is significantly better than existing models.
****

****
**Requirements**
* pytorch 1.8.1
* python  3.8.5
****
**datasets**

Download and unzip training and test data:http://www.bioinf.uni-freiburg.de/Software/GraphProt/GraphProt_CLIP_sequences.tar.bz2
****
**trian**
```
python main.py 
```
**detect motif**
```
python get_motif.py 
```
****           
**Notice**

If FMNet does not converge in your datasets, you can replace Adam with SGD or RMSprop.
