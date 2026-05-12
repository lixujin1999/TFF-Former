# TFF-Former

## 1&nbsp; Installation

Follow the steps below to prepare the virtual environment.

Create and activate the environment:
```shell
conda create -n tsformer python=3.9
conda activate tff_former
```


## 2&nbsp; Experiment

### 2.1&nbsp; Data Acquisition

The data acquisition and preprocessing procedures are consistent for all three tasks. EEG data are recorded using the SynAmp2 system (Neuroscan, Australia) with 64-channel Ag/AgCl electrodes placed according to the international 10/20 system, with a sampling rate of 1000 Hz. All the electrodes with an impedance of 10 kΩ or lower are referenced to the vertex and grounded on the AFz channel.

Wang, Y., Chen, X., Gao, X., & Gao, S. (2016). A benchmark dataset for SSVEP-based brain–computer interfaces. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 25(10), 1746-1752.

### 2.2&nbsp; Data Preprocessing

The collected EEG data are downsampled to 250 Hz. Subsequently, a linear phase 3-order Butterworth filter with a bandpass ranging from 0.5 to 15 Hz is applied to the signal to eliminate slow drift and high-frequency noise and prevent delay distortions. Next, the preprocessed data from each block are segmented into EEG trials. Each trial consists of 1-second EEG data starting from the stimulation onset to 1000 milliseconds (0 s to 1 s) after the stimulation onset. For each trial, data are normalized to zero mean and variance one. The subsequent analysis and classification of EEG signals rely on these segmented EEG trials (samples).


## 3&nbsp; Train

The TSformer-SA is optimized using the two-stage training strategy. The data from existing subjects are first utilized to pretrain the TSformer in advance and the data from new test subject are used to only fine-tune the subjectspecific adapter in the fine-tuning stage.

### 3.1&nbsp; RSVP Task

```bash
python -m torch.distributed.launch --master_port 29502 --nproc_per_node=4 TFF-Former-RSVP/run.py
```
### 3.2&nbsp; SSVEP Task

```bash
python -m torch.distributed.launch --master_port 29502 --nproc_per_node=4 TFF-Former-SSVEP/run.py
```


## 4&nbsp; Cite

If you find this code or our TSformer-SA paper helpful for your research, please cite our paper:

```bibtex
@article{li2025temporal,
  title={A temporal--spectral fusion transformer with subject-specific adapter for enhancing RSVP-BCI decoding},
  author={Li, Xujin and Wei, Wei and Qiu, Shuang and He, Huiguang},
  journal={Neural Networks},
  volume={181},
  pages={106844},
  year={2025},
  publisher={Elsevier}
}
```
