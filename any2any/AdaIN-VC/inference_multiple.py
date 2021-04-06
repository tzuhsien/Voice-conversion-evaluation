import os
import pickle
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm
import yaml
import torch
import torch.nn.functional as F
import soundfile as sf

from models import AE
from preprocess.tacotron.utils import melspectrogram2wav
from preprocess.tacotron.utils import get_spectrograms

def sort_func(element):
    return element[0].shape[0]

def batch_inference(out_mels, vocoder, batch_size):
    out_wavs = []
    for i in tqdm(range(0, len(out_mels), batch_size)):
        right = i + batch_size if len(out_mels) - i >= batch_size else len(out_mels)
        out_wavs.extend(vocoder.generate(out_mels[i:right]))
    return out_wavs

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

        # load testing pairs
        self.pairs = torch.load(args.pairs)

        # load vocoder
        self.vocoder = torch.jit.load(args.vocoder_path).cuda()
        self.use_vocoder = args.use_vocoder

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
        dec = torch.FloatTensor(dec).unsqueeze(0).cuda()
        
        return dec

    def denormalize(self, x):
        m, s = self.attr["mean"], self.attr["std"]
        ret = x * s + m
        return ret

    def normalize(self, x):
        m, s = self.attr["mean"], self.attr["std"]
        ret = (x - m) / s
        return ret

    def write_wav_to_file(self, wav_data, output_path):
        sf.write(output_path, wav_data, self.args.sample_rate)
        return

    def inference_from_path(self):
        conv_mels = []
        if not os.path.exists(os.path.join(self.args.output_dir, 'out_mels.tar')) or not os.path.exists(os.path.join(self.args.output_dir, 'pairs.tar')):
            for pair in tqdm(self.pairs):
                src_mel, _ = get_spectrograms(os.path.join(self.args.source_dir, pair['src_utt']))
                tar_mel, _ = get_spectrograms([os.path.join(self.args.target_dir, i) for i in pair['tgt_utt']], multi=True)
                src_mel = torch.from_numpy(self.normalize(src_mel)).cuda()
                tar_mel = torch.from_numpy(self.normalize(tar_mel)).cuda()
                conv_mel = self.inference_one_utterance(src_mel, tar_mel)
                conv_mels.append(conv_mel.squeeze())
            
            zipped = zip(conv_mels, self.pairs)
            zip_sorted = sorted(zipped, key=sort_func, reverse=True)
            tuples = zip(*zip_sorted)
            conv_mels, self.pairs = [list(i) for i in tuples]
            torch.save(conv_mels, os.path.join(self.args.output_dir, 'out_mels.tar'))
            torch.save(self.pairs, os.path.join(self.args.output_dir, 'pairs.tar'))
        else:
            conv_mels = torch.load(os.path.join(self.args.output_dir, 'out_mels.tar'))
            self.pairs = torch.load(os.path.join(self.args.output_dir, 'pairs.tar'))
        if self.use_vocoder == False:
            return
        del self.model
        with torch.no_grad():
            wav_datas = batch_inference(conv_mels, self.vocoder, batch_size=21)

        for i, pair in tqdm(enumerate(self.pairs)):
            wav_data = wav_datas[i].cpu().numpy()
            prefix = Path(pair['src_utt']).name.replace('/wav/', '_').replace('.wav', '') 
            postfix = Path(pair['tgt_utt'][0]).name.replace('/wav/', '_').replace('.wav', '')
            # prefix = pair['src_utt'].replace('.wav', '').split('/')[1]
            self.write_wav_to_file(wav_data, os.path.join(self.args.output_dir, f"{prefix}_to_{postfix}.wav"))

        return


def main():
    parser = ArgumentParser()
    parser.add_argument("--attr", default="checkpoints/attr.pkl")
    parser.add_argument("--config", default="checkpoints/config.yaml")
    parser.add_argument("--model", default="checkpoints/vctk_model.ckpt")
    parser.add_argument("--pairs", help="testing pair path")
    parser.add_argument("-s", "--source_dir", help="source path")
    parser.add_argument("-t", "--target_dir", help="target path")
    parser.add_argument("-o", "--output_dir", help="output path")
    parser.add_argument("-v", "--vocoder_path", help="vocoder path", default='checkpoints/parallel-vocoder-ckpt-150000.pt')
    parser.add_argument("--use_vocoder", type=bool, default=True)
    parser.add_argument("--sample_rate", default=24000, type=int)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.config) as f:
        config = yaml.load(f)

    inferencer = Inferencer(config=config, args=args)
    inferencer.inference_from_path()


if __name__ == "__main__":
    main()
