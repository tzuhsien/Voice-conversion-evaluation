import pickle
from argparse import ArgumentParser

import yaml
import torch
import torch.nn.functional as F
from scipy.io.wavfile import write

from models import AE
from preprocess.tacotron.utils import melspectrogram2wav
from preprocess.tacotron.utils import get_spectrograms


class Inferencer:
    def __init__(self, config, args):
        # config store the value of hyperparameters, turn to attr by AttrDict
        self.config = config
        print(config)
        # args store other information
        self.args = args
        print(self.args)

        # init the model with config
        self.build_model()

        # load model
        self.load_model()

        with open(self.args.attr, "rb") as f:
            self.attr = pickle.load(f)

    def load_model(self):
        print(f"Load model from {self.args.model}")
        self.model.load_state_dict(torch.load(f"{self.args.model}"))
        return

    def build_model(self):
        # create model, discriminator, optimizers
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AE(self.config).to(device)
        print(self.model)
        self.model.eval()
        return

    def utt_make_frames(self, x):
        frame_size = self.config["data_loader"]["frame_size"]
        remains = x.size(0) % frame_size
        if remains != 0:
            x = F.pad(x, (0, remains))
        out = x.view(1, x.size(0) // frame_size, frame_size * x.size(1)).transpose(1, 2)
        return out

    def inference_one_utterance(self, x, x_cond):
        x = self.utt_make_frames(x)
        x_cond = self.utt_make_frames(x_cond)
        dec = self.model.inference(x, x_cond)
        dec = dec.transpose(1, 2).squeeze(0)
        dec = dec.detach().cpu().numpy()
        dec = self.denormalize(dec)
        wav_data = melspectrogram2wav(dec)
        return wav_data, dec

    def denormalize(self, x):
        m, s = self.attr["mean"], self.attr["std"]
        ret = x * s + m
        return ret

    def normalize(self, x):
        m, s = self.attr["mean"], self.attr["std"]
        ret = (x - m) / s
        return ret

    def write_wav_to_file(self, wav_data, output_path):
        write(output_path, rate=self.args.sample_rate, data=wav_data)
        return

    def inference_from_path(self):
        src_mel, _ = get_spectrograms(self.args.source)
        tar_mel, _ = get_spectrograms(self.args.target)
        src_mel = torch.from_numpy(self.normalize(src_mel)).cuda()
        tar_mel = torch.from_numpy(self.normalize(tar_mel)).cuda()
        conv_wav, conv_mel = self.inference_one_utterance(src_mel, tar_mel)
        self.write_wav_to_file(conv_wav, self.args.output)
        return


def main():
    parser = ArgumentParser()
    parser.add_argument("--attr", default="checkpoints/attr.pkl")
    parser.add_argument("--config", default="checkpoints/config.yaml")
    parser.add_argument("--model", default="checkpoints/vctk_model.ckpt")
    parser.add_argument("-s", "--source", help="source wav path")
    parser.add_argument("-t", "--target", help="target wav path")
    parser.add_argument("-o", "--output", help="output wav path")
    parser.add_argument("--sample_rate", default=24000, type=int)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)

    inferencer = Inferencer(config=config, args=args)
    inferencer.inference_from_path()


if __name__ == "__main__":
    main()
