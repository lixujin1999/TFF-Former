# TFF-Former

## 1&nbsp; Installation

Follow the steps below to prepare the virtual environment.

Create and activate the environment:
```shell
conda create -n tff_former python=3.9
conda activate tff_former
```


## 2&nbsp; Experiment

### 2.1&nbsp; Data Acquisition

The proposed TFF-Former are validated on two EEG datasets: 1 ) self-collected RSVP dataset. 2 ) the benchmark dataset for SSVEP tasks. 

RSVP Dataset: The experiment included 31 participants (19 males and 12 females; aged 24.9 ± 2.8, 28 right-handed). The visual stimuli for our experiment included 1,400 images (500×500 pixels) from the scene and object database published by MIT CSAIL. These images were divided into target images with pedestrians and non-target images without pedestrians. Images were randomly presented at a frequency of 10 Hz, where the probability of the target image appearance was 4%. Each experimental session had 10 blocks, and each block contained 1400 images, divided into 14 sequences.

Benchmark Dataset: This SSVEP dataset has 35 subjects (17 females, aged 17-34 years), including 40 targets. The 40 targets were  coded using the JFPM method. The frequencies range from 8 Hz to 15.8 Hz with an interval of 0.2 Hz, and the phase difference between two adjacent targets was 0.5 𝜋. For each subject, the experiment included 6 blocks, and each block contained 40 trials corresponding to all targets indicated once in random order. In the public dataset, the trial length of 6 s includes 0.5 s before stimulus onset, 5 s for stimulation, and 0.5 s after stimulus offset.

### 2.2&nbsp; Data Preprocessing

In the preprocessing stage, the RSVP dataset were down-sampled to 250 Hz. After that, a linear phase 3-order ButterWorth filter with a bandpass between 0.5 and 15 Hz is used to filter the signal to remove slow drift and high-frequency noise and prevent delay distortions. Then the preprocessed data of each block were segmented into EEG trials each containing 1 second EEG data. For each trial, data was normalized to zero mean andvariance one. The subsequent analysis and classification of EEG were based on these segmented EEG trials (samples). According to our experimental paradigm, each subject had 10 (blocks) ×1400 (trials) EEG samples per session, where 560 are target samples and the rest are non-target samples. The SSVEP recordings were passed through a Chebyshev Type I band-pass filter with the range of 8 Hz to 90 Hz. We applied a notch filter at 50 Hz to remove the common powerline noise and also normalized each trial to zero mean and variance one. We used data from 0.64 to 2.64 seconds in each trial which contains 500 sampling points.

## 3&nbsp; Train

The TSformer-SA is trained by minimizing the cross-entropy loss function. Adam optimizer is adopted for model optimization and the learning rate is 0.0005 in RSVP task and 0.001 in SSVEP task with a 20% decrease every 40 epochs. The L2 regularization is adopted, and the weight decay coefficient is 0.01. In SSVEP task we also used label smoothing regularization with 𝜖 = 0.005. The batch size is set to 64 and the maximum number of training epochs is set to 100.

### 3.1&nbsp; RSVP Task

```bash
python -m torch.distributed.launch --master_port 29502 --nproc_per_node=4 TFF-Former-RSVP/run.py
```
### 3.2&nbsp; SSVEP Task

```bash
python -m torch.distributed.launch --master_port 29502 --nproc_per_node=4 TFF-Former-SSVEP/run.py
```


## 4&nbsp; Cite

If you find this code or our TFF-Former paper helpful for your research, please cite our paper:

```bibtex
@inproceedings{li2022tff,
  title={TFF-Former: Temporal-frequency fusion transformer for zero-training decoding of two BCI tasks},
  author={Li, Xujin and Wei, Wei and Qiu, Shuang and He, Huiguang},
  booktitle={Proceedings of the 30th ACM international conference on multimedia},
  pages={51--59},
  year={2022}
}
```
