All audio paths in metadata are relative paths.

For reducing the inference time, the pairs in metadata are sorted by the lengths of source audios from long to short.

The metadata contains:
- source_corpus: The name of the source dataset.
- source_corpus_speaker_number: The number of speaker in source dataset.
- source_random_seed: Random seed used for sampling source utterance.
- target_corpus: The name of the target dataset.
- target_corpus_speaker_number: The number of speaker in target dataset.
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