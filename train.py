from dataclasses import asdict, dataclass
from functools import cached_property
from itertools import islice
import math
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import lightning as lit

from lightning.pytorch.cli import LightningCLI
from torch.utils.data import Dataset, DataLoader
from typing import NamedTuple
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from einops.layers.torch import Rearrange
from x_transformers import Decoder


torch.set_float32_matmul_precision('medium')


class LangugeModellingDataset(Dataset):
    def __init__(
        self,
        text: str,
        length: int | tuple[int, int],
        random_sample: bool = False,
        num_samples: int | None = None,
    ):
        self.text = text
        self.length = length
        self.random_sample = random_sample
        self.num_samples = num_samples

        assert (
            not random_sample or num_samples is not None
        ), "Random sampling requires num_samples to be set"

        if isinstance(length, int):
            self.length_range = (length, length + 1)
        elif isinstance(length, (tuple, list)):
            assert len(length) == 2

            min, max = length

            if max != min + 1:
                assert random_sample and num_samples is not None

                if random_sample and (min + 1) < max and num_samples is None:
                    raise ValueError("Random sampling requires num_samples to be set")

            self.length_range = (min, max)
        else:
            raise ValueError

    def split(
        self, fraction: float | int, train_kwargs: dict = {}, val_kwargs: dict = {}
    ):
        if isinstance(fraction, float):
            split_idx = int(len(self.text) * (1 - fraction))
        else:
            split_idx = len(self.text) - fraction

        kwargs = dict(
            length=self.length,
            random_sample=self.random_sample,
            num_samples=self.num_samples,
        )

        return (
            LangugeModellingDataset(
                self.text[:split_idx], **{**kwargs, **train_kwargs}
            ),
            LangugeModellingDataset(self.text[split_idx:], **{**kwargs, **val_kwargs}),
        )

    def __len__(self):
        if self.random_sample:
            return self.num_samples

        return math.ceil(
            (len(self.text) - self.length_range[0] - 1) / self.length_range[0]
        )

    def __getitem__(self, idx):
        if self.random_sample:
            length = random.randint(*self.length_range)
            start = random.randint(0, len(self.text) - length - 1)
            return self.text[start : start + length], self.text[
                start + 1 : start + length + 1
            ]

        length = self.length_range[0]

        start = idx * length
        end = start + length
        return self.text[start:end], self.text[start + 1 : end + 1]


class ChunkedGPTOutput(NamedTuple):
    group_logits: torch.Tensor
    target_indices: torch.Tensor


class ChunkedGPT(nn.Module):
    def __init__(
        self, vocab_size, group_size, embed_dim, dim, depth, heads, pad_idx, **kwargs
    ):
        super().__init__()

        assert group_size > 0

        self.vocab_size = vocab_size
        self.group_size = group_size
        self.dim = dim
        self.pad_idx = pad_idx
        self.depth = depth
        self.heads = heads

        self.group_tokens = Rearrange("b (n g) -> b n g", g=self.group_size)
        self.inflate_groups = Rearrange("b n g d -> b n (g d)", g=self.group_size)

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.group_proj = nn.Linear(embed_dim * group_size, dim)

        self.decoder = Decoder(dim=dim, depth=depth, heads=heads, **kwargs)

        self.to_logits = nn.Linear(dim, vocab_size)

    def validate_input(self, x):
        if torch.any(x[:, -1] == self.pad_idx):
            raise AssertionError(
                "ChunkedGPT expects prepadding but found at least one padding token "
                "in the last position of the input which indicates postpadding."
            )

    def prepare_input(self, x):
        b, n = x.shape

        target_indices = (
            torch.arange(math.ceil(n / self.group_size), device=x.device) + 1
        ) * self.group_size

        num_pad = (self.group_size - n % self.group_size) % self.group_size

        if self.group_size > 1 and num_pad > 0:
            x = torch.cat(
                [
                    torch.full(
                        (b, num_pad),
                        fill_value=self.pad_idx,
                        device=x.device,
                        dtype=x.dtype,
                    ),
                    x,
                ],
                dim=1,
            )

            target_indices -= num_pad

        token_groups = self.group_tokens(x)
        group_masking = (token_groups == self.pad_idx).sum(-1) != self.group_size

        return token_groups, group_masking, target_indices

    def forward(self, x):
        b, n = x.shape

        self.validate_input(x)

        token_groups, group_masking, target_indices = self.prepare_input(x)
        token_embeds = self.embed(token_groups)

        group_embeds = self.inflate_groups(token_embeds)

        group_input = self.group_proj(group_embeds)
        group_output = self.decoder(group_input, mask=group_masking)
        group_logits = self.to_logits(group_output)

        return ChunkedGPTOutput(group_logits, target_indices)

    def forward_all_targets(self, x):
        b, n = x.shape

        logits = torch.empty(
            b, n + 1, self.vocab_size, device=x.device, dtype=torch.float32
        )

        for i in range(self.group_size):
            group_logits, target_indices = self(x)

            logits[:, target_indices] = group_logits.type(logits.dtype)

            x = x[:, :-1]

        return logits[:, 1:]

    @torch.no_grad()
    def generate(self, input):
        if input.ndim == 1:
            input = input.unsqueeze(0)

        assert input.shape[0] == 1, "only supports generation for single sequence"

        while True:
            group_logits, _ = self(input)

            logits = group_logits[:, -1]

            # "Mute" the pad token
            logits[:, self.pad_idx] = -math.inf

            probas = F.softmax(logits, dim=-1)

            next_token = torch.multinomial(probas, 1)

            input = torch.cat([input, next_token], dim=-1)

            yield next_token.item()


class CosineWithWarmupLR(LambdaLR):
    def __init__(
        self,
        optimizer: Optimizer,
        training_steps: int,
        warmup_steps: int = 0,
        num_cycles: float = 0.5,
    ):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(
                max(1, training_steps - warmup_steps)
            )
            return max(
                0.0,
                0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
            )

        super().__init__(optimizer, lr_lambda, -1)


@dataclass
class ModelConfig:
    embed_dim: int = 32
    dim: int = 128
    depth: int = 4
    heads: int = 8

    group_size: int = 1


@dataclass
class DataConfig:
    path: str
    num_val_symbols: int = 5_000_000
    num_test_symbols: int = 5_00_0000
    train_batch_size: int = 64
    eval_batch_size: int | None = None
    train_batches_per_epoch: int = 1_000
    train_length: int = 512
    eval_length: int = 512
    num_workers: int = 4

    @property
    def eval_symbols(self):
        return self.num_val_symbols + self.num_test_symbols


@dataclass
class TrainingConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 1000


class LitChunkedGPT(lit.LightningModule):
    def __init__(
        self,
        model_config: ModelConfig,
        data_config: DataConfig,
        training_config: TrainingConfig,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model_config = model_config
        self.data_config = data_config
        self.training_config = training_config

        self.model = None
        self.dataset = None
        self.splits = None
        self.vocab = None

    @cached_property
    def i2b(self):
        return self.vocab

    @cached_property
    def b2i(self):
        return {b: i for i, b in enumerate(self.i2b)}
    
    @property
    def pad_idx(self):
        return len(self.vocab)

    def collate_fn(self, items):
        source_items, target_items = zip(*items)

        pad_kwargs = dict(batch_first=True, padding_value=self.pad_idx, padding_side="left")

        inputs = pad_sequence(
            [torch.tensor(item) for item in source_items], **pad_kwargs
        )
        targets = pad_sequence(
            [torch.tensor(item) for item in target_items], **pad_kwargs
        )

        return inputs, targets

    def train_dataloader(self):
        return DataLoader(
            self.splits[0],
            batch_size=self.data_config.train_batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.data_config.num_workers,
        )

    def eval_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.data_config.eval_batch_size or self.data_config.train_batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.data_config.num_workers,
        )

    def val_dataloader(self):
        return self.eval_dataloader(self.splits[1])

    def test_dataloader(self):
        return self.eval_dataloader(self.splits[2])
    
    def prepare_data(self):
        subprocess.run(["bash", "download_data.sh"])

    def prepare_dataset(self):
        train_length = self.data_config.train_length * self.model_config.group_size
        eval_length = self.data_config.eval_length * self.model_config.group_size
        
        symbols = np.fromfile(self.data_config.path, dtype=np.uint8)
        self.vocab, symbols = np.unique(symbols, return_inverse=True)

        self.dataset = LangugeModellingDataset(
            symbols, length=eval_length
        )
        
        group_size = self.model_config.group_size

        train_split, eval_split = self.dataset.split(
            self.data_config.eval_symbols,
            train_kwargs=dict(
                length=(train_length, train_length + group_size - 1),
                random_sample=True,
                num_samples=self.data_config.train_batches_per_epoch
                * self.data_config.train_batch_size,
            ),
        )

        val_split, test_split = eval_split.split(
            self.data_config.num_test_symbols,
        )

        assert len(val_split.text) == self.data_config.num_val_symbols
        assert len(test_split.text) == self.data_config.num_test_symbols

        self.splits = train_split, val_split, test_split

    def prepare_model(self):
        self.model = ChunkedGPT(
            vocab_size=len(self.vocab) + 1,
            **asdict(self.model_config),
            pad_idx=self.pad_idx,
            rotary_pos_emb=True,
        )
    def setup(self, stage: str):
        self.prepare_dataset()
        self.prepare_model()

    def encode_text(self, text):
        return torch.tensor([self.b2i[ord(char)] for char in text])

    def decode_tokens(self, tensor):
        return "".join(chr(self.i2b[int(idx)]) for idx in tensor)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.training_config.lr,
            weight_decay=self.training_config.weight_decay,
        )

        scheduler = CosineWithWarmupLR(
            opt,
            len(self.train_dataloader()) * self.trainer.max_epochs,
            warmup_steps=self.training_config.warmup_steps,
        )

        return [opt], [
            {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        ]

    def build_log_dict(self, loss, prefix):
        return {
            f"{prefix}/loss": loss,
            f"{prefix}/ppl": math.exp(loss),
            f"{prefix}/bpc": loss / math.log(2),
        }
        
    def training_step(self, batch, batch_idx, log_prefix="train"):
        inputs, targets = batch

        group_logits, group_indices = self.model(inputs)

        target_indices = group_indices - 1

        group_targets = targets[:, target_indices]

        loss = F.cross_entropy(
            group_logits.transpose(1, 2), group_targets, ignore_index=self.pad_idx
        )

        self.log_dict(self.build_log_dict(loss, log_prefix), prog_bar=True, sync_dist=True)

        return loss

    def eval_step(self, batch, batch_idx, log_prefix="eval"):
        inputs, targets = batch

        logits = self.model.forward_all_targets(inputs)

        loss = F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=self.pad_idx)

        self.log_dict(self.build_log_dict(loss, log_prefix), prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, log_prefix="val")

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch, batch_idx, log_prefix="test")
    
    def on_validation_epoch_end(self):
        if not self.trainer.is_global_zero:
            return
        
        prompt = "Deep Learning is a "
        
        print()
        print(prompt, end="")
        
        for token in islice(self.model.generate(self.encode_text(prompt).to(self.device)), 512):
            print(chr(self.i2b[token]), end="", flush=True)
        
        print()
        


def main():
    cli = LightningCLI(LitChunkedGPT, save_config_callback=None)


if __name__ == "__main__":
    main()
