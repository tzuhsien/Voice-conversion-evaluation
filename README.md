# Voice-conversion-evaluation
An evaluation toolkit for voice conversion models.

# Metrics
The metrics contain mean score opinion, character error rate, and speaker recognition accept rate.
- Mean score opinion
  - Ensemble several MBnet which is implemented by [sky1456723](https://github.com/sky1456723/Pytorch-MBNet).
```
  python calculate_objective_metric.py -d [data_dir] -r metrics/mean_opinion_score
```
- Character error rate:
  - Use the automatic speech recognition model on [hugging face](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self).
  - The word error rate on Librispeech test-other is 3.9.
```
  python calculate_objective_metric.py -d [data_dir] -r metrics/character_error_rate
```
- Speaker recognition accept rate:
  - You can calculate the threshold by ```metrics/speaker_verification/equal_error_rate/```.
  - And some thresholds are in ``` metrics/speaker_verification/equal_error_rate/threshold.yaml```.
```
  python calculate_objective_metric.py -d [data_dir] -r metrics/speaker_verification -t [target_dir] -th [threshold path]
```
