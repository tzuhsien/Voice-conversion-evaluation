"""Inferencer of AdaIN-VC"""
from pathlib import Path
import yaml
import joblib
import torch
import torch.nn.functional as F

from .model import AE
from .audioprocessor import AudioProcessor


class Inferencer:
    """Inferencer"""

    def __init__(self, root):
        checkpoint_dir = Path(root) / "checkpoints"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config_path = checkpoint_dir / "config.yaml"
        model_path = checkpoint_dir / "model.ckpt"
        attr_path = checkpoint_dir / "attr.pkl"
        config = yaml.load(config_path.open(), Loader=yaml.FullLoader)

        self.model = AE(config).to(device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        self.attr = joblib.load(attr_path)

        self.device = device
        self.frame_size = config["data_loader"]["frame_size"]

    def denormalize(self, d_mel):
        """denormalize"""
        mean, std = self.attr["mean"], self.attr["std"]
        ret = d_mel * std + mean

        return ret

    def normalize(self, d_mel):
        """normalize"""
        mean, std = self.attr["mean"], self.attr["std"]
        ret = (d_mel - mean) / std

        return ret

    def preprocess(self, d_mel):
        """Preprocess"""
        d_mel = torch.from_numpy(self.normalize(d_mel)).to(self.device)

        remains = d_mel.size(0) % self.frame_size
        if remains != 0:
            d_mel = F.pad(d_mel, (0, remains))

        d_mel = d_mel.view(
            -1, d_mel.size(0) // self.frame_size, self.frame_size * d_mel.size(1)
        ).transpose(1, 2)

        return d_mel

    def inference(self, src_mel, tgt_mels):
        """Inference one utterance."""
        src_mel = self.preprocess(src_mel)
        tgt_mels = [self.preprocess(tgt_mel) for tgt_mel in tgt_mels]

        with torch.no_grad():
            latent = self.model.encode_linguistic(src_mel)
            tgt_embs = [self.model.encode_timbre(tgt_mel) for tgt_mel in tgt_mels]
            tgt_emb = torch.mean(torch.cat(tgt_embs), 0, keepdim=True)
            dec = self.model.decode(latent, tgt_emb)
            dec = dec.transpose(1, 2).squeeze(0)
            dec = dec.detach().cpu().numpy()
            dec = self.denormalize(dec)

        return torch.from_numpy(dec)

    def inference_from_file(self, source, targets):
        """Inference from file path."""
        src_mel = AudioProcessor.get_input(source)
        tgt_mels = [AudioProcessor.get_input(target) for target in targets]

        result = self.inference(src_mel, tgt_mels)

        return result
