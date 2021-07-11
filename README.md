# Voice-conversion-evaluation
An evaluation toolkit for voice conversion models. \

# Make metadata
Generate the metadata for evaluating models. \
You can find several dataset parsers in the directory of parsers.
There is an example for generating metadata below.
```
  python make_metadata.py VCTK /path_of_datasets/VCTK-Corpus CMU /path_of_datasets/CMU_ARCTIC -n 10 -nt 5 -o [path of output dir]
```
You can find an example metadata in the directory of examples. \ 
The pairs in metadata are sorted by the lengths of source audios from long to short. \
All file paths in metadata are relative path. \
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
  - src_utt: The relative path of the source utterance
  - tgt_utts: List of the relative path of target utterances
  - content: The content of the source utterance.
  - src_second: The second of the source utterance.

# Inference
Easy to inference several voice conversion models with  a unified I/O interface.
There is an example for inference below.
```
  python inference.py -m examples/metadata_example.json -s /path_of_datasets/VCTK-Corpus -t /path_of_datasets/CMU_ARCTIC -o [path of output dir] -r models/any2any/AdaIN-VC
```

# Metrics
The metrics include Nerual mean opinion score assessment, character error rate, and speaker verification acceptance rate.
- Nerual mean opinion score assessment:
  - Ensemble several MBNet which is implemented by [sky1456723](https://github.com/sky1456723/Pytorch-MBNet).
  - You can calculate nerual mean opinion score assessment without metadata.
  ```
    python calculate_objective_metric.py -d [data_dir] -r metrics/mean_opinion_score
  ```
- Character error rate:
  - Use the automatic speech recognition model provided by [Hugging Face](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self).
  - You should prepare metadata before you calculate character error rate.
  ```
    python calculate_objective_metric.py -d [data_dir] -r metrics/character_error_rate
  ```
- Speaker verification acceptance rate:
  - You can calculate the equal error rate and threshold by ```metrics/speaker_verification/equal_error_rate/```.
  - And some pre-calculated thresholds are in ``` metrics/speaker_verification/equal_error_rate/```.
  - You should prepare metadata before you speaker verification acceptance rate.
  ```
    python calculate_objective_metric.py -d [data_dir] -r metrics/speaker_verification -t [target_dir] -th [threshold_path]
  ```
