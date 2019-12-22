#Disentangling time-variant and time-invariant factors for improved classification of EEG signals

##Abstract: 
Two main challenges in classifying stimuli using EEG signals are 1) Low signal-to-noise ratio. This is due to the time-variant factors appearing in the course of measurements. These time-varying factors can be electrical surroundings, muscle activity, eye movements of blinks, etc. 2) Variabilities between individual subjects.
To this end, we propose a novel architecture based on the recent development of disentangled representation and probabilistic sequential modeling. The underlying architecture is a Conv1dLSTM, that utilizes only the invariant factors for classification. We hoped that disentangling time-varying and time-invariant dynamics apparent in the sequence of EEG data, increase the classification accuracy. Our experiment using MIIR dataset shows we can achieve accuracy of 21.67\% in test time, verified using the outer 9-fold cross-validation performed across subjects as in [1](http://bib.sebastianstober.de/icassp2017.pdf).

##Dataset:
The Mirr dataset contains 64 EEG channels, 9 subjects and 12 audio stimuli for 540 trails. Measurements sequences of are length 3518. They have been normalized to zero-mean and range[-1,1]. Therefore no normalization/zfiltering is necessary.

##ML considerations:
The seq length of 3518 is way longer than the 250-300 steps used in practice for LSTM. We, therefore, first apply a Conv1d with a kernel size of 320 and stride 160 to reduce the length of sequences to 20. We used factored disentangled representation for sequential data, described in the paper [2](https://arxiv.org/pdf/1803.02991.pdf). Using similar techniques presented in [3]((https://openreview.net/pdf?id=Sy2fzU9gl)), [2](https://arxiv.org/pdf/1803.02991.pdf) derives time-variant encodings z and time-invariant features f for sequential data. Our architecture has 2 main differences. 1) First, we are concerned with classification rather than data generation. The decoder is, therefore, is replaced the decoder with a classifier and the reconstruction loss with CrossEntropy loss 2) Most importantly, unlike [2](https://arxiv.org/pdf/1803.02991.pdf) where (z, f) is passed to the decoder, we only use f to output classifications. This is to ignore that time-variant factor/noises appearing in the course of experiments.


##Training and Evaluation scheme
Verification is being conducted using the outer 9-fold cross-validation performed across subjects as in [1](http://bib.sebastianstober.de/icassp2017.pdf). A random subject is excluded from the training and the rest of the data get used for the training. The data for the excluded subject then gets used for validation.

## Possible future improvements:
For now, we consider the prior of Beat EPRS (event_related_potentials) to be Gaussian. An ERP is an electrophysiological response that occurs as a direct result of a stimulus. In the current MIIR case is an audio stimulus. Interestingly, in both cases of EEG signal and auditory data, prior over the time-invariant encoder are better to be distributions with higher kurtosis than the Gaussian such as Laplacian. I did a small experiment but haven't made it work yet.

To repeat the experiment:

Download the Mirr data in .h5 format from [here](http://www.ling.uni-potsdam.de/mlcog/OpenMIIR/rl2016/data/) and place it  on a directory named data.

```
python3 trainer.py
```

##Acknowledgments:

https://openmiir.github.io/

https://github.com/yatindandi/Disentangled-Sequential-Autoencoder
