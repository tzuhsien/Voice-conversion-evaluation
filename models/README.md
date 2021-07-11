# Voice conversion models
Voice conversion models are classified by the number of source speakers and the number of target speakers.

There are only inference codes in this repo. If you want to train the model by yourself, please to visit the official repo.

If you want to add new voice conversion models, you need finish the ```audioprocessor.py``` and ```inferencer.py``` by yourself.
- audioprocessor: Process audio data, include ```load_wav```, ```wav2spectrogram```, and ```file2spectrogram```.
- inferencer: Convert audio and generate the raw waveform, include ```inference_from_pair``` and ```spectrogram2waveform```

# Reference
- Any2any
    - [AdaIN-VC](https://github.com/jjery2243542/adaptive_voice_conversion)
    - [AutoVC](https://github.com/auspicious3000/autovc)
    - [VQVC+](https://github.com/ericwudayi/SkipVQVC)
    - [FragmentVC](https://github.com/yistLin/FragmentVC)
- many2many
    - [DGAN-VC](https://github.com/jjery2243542/voice_conversion)
    - [BLOW](https://github.com/joansj/blow)
    - [WAStarGAN-VC](https://github.com/MingjieChen/LowResourceVC)