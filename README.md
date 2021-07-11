# Voice-conversion-evaluation
An evaluation toolkit for voice conversion models.

# Make metadata
The metadata plays an important role in this repo. There are several information in the metadata, include dataset name, number of samples, speaker names, the relative path of audio, and the content of source audio. 

You can find more information with metadata [here](https://github.com/tzuhsien/Voice-conversion-evaluation/blob/master/examples/README.md).

There is an example for generating metadata.
```
  python make_metadata.py \ 
    VCTK /path_of_datasets/VCTK-Corpus \ 
    CMU /path_of_datasets/CMU_ARCTIC \ 
    -n 10 -nt 5 \ 
    -o [path of output dir]
```
You can find an example metadata in the directory of examples.

We provide several dataset parsers in the directory of parsers. The default parser is the same as the dataset name. You can name the dataset by yourself and specify a particular parser.

# Inference
Utilize a unified I/O interface to inference voice conversion models. You should prepare metadata before you inference voice conversion models. 

After inferencing voice conversion models, the relative path of converted audios will be add into the metadata, and the metadata will be copied into the output directory. 

All of the pretrained models can be found in [releases](https://github.com/tzuhsien/Voice-conversion-evaluation/releases). Please put the checkpoints into the corresponding model directory, e.g. ```models/any2any/AdaIN-VC/checkpoints```.

There is an example for inference.
```
  python inference.py \ 
    -m examples/metadata_example.json \ 
    -s /path_of_datasets/VCTK-Corpus \ 
    -t /path_of_datasets/CMU_ARCTIC \ 
    -o [path of output dir] \ 
    -r models/any2any/AdaIN-VC
```

For BLOW, there are some issues when reloading the checkpoint. Please inference BLOW in its directory.

# Metrics
The metrics include Nerual mean opinion score assessment, character error rate, and speaker verification acceptance rate.

If you only want to use the metrics here, it is unnecessary to use the inference code in this repo. You can use your own inference code with the metadata. Be careful, you need to add the relative paths of converted audios into metadata.

- Nerual mean opinion score assessment:
  - Ensemble several MBNet which is implemented by [sky1456723](https://github.com/sky1456723/Pytorch-MBNet).
  - You can calculate nerual mean opinion score assessment without metadata.
  ```
    python calculate_objective_metric.py \ 
      -d [data_dir] \ 
      -r metrics/mean_opinion_score
  ```
- Character error rate:
  - Use the automatic speech recognition model provided by [Hugging Face](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self).
  - You should prepare metadata before you calculate character error rate.
  ```
    python calculate_objective_metric.py \ 
      -d [data_dir] \ 
      -r metrics/character_error_rate
  ```
- Speaker verification acceptance rate:
  - Use the speaker verification model provided by [Resemblyzer](https://github.com/resemble-ai/Resemblyzer).
  - You can calculate the equal error rate and threshold by ```metrics/speaker_verification/equal_error_rate/```.
  - And some pre-calculated thresholds are in ``` metrics/speaker_verification/equal_error_rate/```.
  - You should prepare metadata before you calculate speaker verification acceptance rate.
  ```
    python calculate_objective_metric.py \ 
      -d [data_dir] \ 
      -r metrics/speaker_verification \ 
      -t [target_dir] \ 
      -th [threshold_path]
  ```

# Reference Repositories
## Voice conversion models
- [AdaIN-VC](https://github.com/jjery2243542/adaptive_voice_conversion)
- [AutoVC](https://github.com/auspicious3000/autovc)
- [VQVC+](https://github.com/ericwudayi/SkipVQVC)
- [FragmentVC](https://github.com/yistLin/FragmentVC)
- [DGAN-VC](https://github.com/jjery2243542/voice_conversion)
- [BLOW](https://github.com/joansj/blow)
- [WAStarGAN-VC](https://github.com/MingjieChen/LowResourceVC)

## Metrics
- [MBNet](https://github.com/sky1456723/Pytorch-MBNet)
- [Resemblyzer](https://github.com/resemble-ai/Resemblyzer)
- [Hugging Face](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self)

## Others
- [Vocoder](https://github.com/yistLin/universal-vocoder)