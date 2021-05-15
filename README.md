# Voice-conversion-evaluation
An evaluation toolkit for voice conversion models.

# Sample test pair
Generate the metadata for evaluating models. \
The directory of parsers contains several available corpus parsers.
```
  python sampler.py [name of source corpus] [path of source dir] [name of target corpus] [path of target dir] -n [number of samples] -nt [number of target utterances] -o [path of output dir]
```
The pairs of metadata are sorted by src_second for long to short. \
The metadata contains:
- source_corpus: The name of the source corpus.
- source_corpus_speaker_number: The number of speaker in source corpus.
- source_random_seed: Random seed used for sampling source utterance.
- target_corpus: The name of the target corpus.
- target_corpus_speaker_number: The number of speaker in target corpus.
- target_random_seed: Random seed used for sampling target utterances.
- n_samples: number of samples
- n_target_samples: number of target utterances
- pairs: List of evaluating pairs
  - source_speaker: The name of the source speaker.
  - target_speaker: The name of the target speaker.
  - src_utt: The relative path of the source utterance, which is relative to the source dir.
  - tgt_utts: List of the relative path of target utterances, which is relative to the target dir.
  - content: The content of the source utterance.
  - src_second: The second of the source utterance.
  - converted: The entry does not appear when use sampler, you need to add the relative path for your converted output.

# Metrics
The metrics include automatic mean opinion score assessment, character error rate, and speaker verification acceptance rate.
- Automatic mean opinion score assessment
  - Ensemble several MBNet which is implemented by [sky1456723](https://github.com/sky1456723/Pytorch-MBNet).
  ```
    python calculate_objective_metric.py -d [data_dir] -r metrics/mean_opinion_score
  ```
- Character error rate:
  - Use the automatic speech recognition model provided by [Hugging Face](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self).
  - The word error rate on Librispeech test-other is 3.9.
  ```
    python calculate_objective_metric.py -d [data_dir] -r metrics/character_error_rate
  ```
- Speaker verification acceptance rate:
  - You can calculate the threshold by ```metrics/speaker_verification/equal_error_rate/```.
  - And some pre-calculated thresholds are in ``` metrics/speaker_verification/equal_error_rate/threshold.yaml```.
  ```
    python calculate_objective_metric.py -d [data_dir] -r metrics/speaker_verification -t [target_dir] -th [threshold path]
  ```
