# scgpt_embed.py — Standalone scGPT cell embedding extractor
"""
Standalone scGPT cell encoder that does NOT depend on the scgpt pip package.
Instead, it directly loads the TransformerModel weights and vocab.json,
avoiding the torchtext compatibility issues with PyTorch nightly.

This module reads the real scGPT architecture (bowang-lab/scGPT) and
reimplements only the cell embedding extraction path.

Usage:
    encoder = ScGPTCellEncoder("models/scgpt_human")
    embeddings = encoder.encode(adata)  # returns (N, 512) np.ndarray
"""

import json
import logging
from pathlib import Path
from typing import Union, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ============================================================================
#  Minimal GeneVocab (replaces torchtext dependency)
# ============================================================================

class SimpleGeneVocab:
    """Minimal gene vocabulary: JSON dict of {gene_name: int_id}.

    Replaces scGPT's GeneVocab which depends on torchtext.
    """

    def __init__(self, token2idx: dict):
        self._token2idx = token2idx
        self._idx2token = {v: k for k, v in token2idx.items()}
        self._default_idx = None

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "SimpleGeneVocab":
        with open(path, "r") as f:
            token2idx = json.load(f)
        return cls(token2idx)

    def __getitem__(self, token: str) -> int:
        if token in self._token2idx:
            return self._token2idx[token]
        if self._default_idx is not None:
            return self._default_idx
        raise KeyError(f"Token '{token}' not in vocabulary")

    def __contains__(self, token: str) -> bool:
        return token in self._token2idx

    def __len__(self) -> int:
        return len(self._token2idx)

    def __call__(self, tokens: list) -> list:
        return [self[t] for t in tokens]

    def set_default_index(self, idx: int):
        self._default_idx = idx

    def append_token(self, token: str):
        if token not in self._token2idx:
            idx = max(self._token2idx.values()) + 1
            self._token2idx[token] = idx
            self._idx2token[idx] = token

    def get_stoi(self) -> dict:
        return self._token2idx


# ============================================================================
#  Minimal scGPT Model Components (from bowang-lab/scGPT source)
# ============================================================================

class GeneEncoder(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx=None):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        return self.enc_norm(self.embedding(x))


class ContinuousValueEncoder(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_value: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x)


class CategoryValueEncoder(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx=None):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = x.long()
        return self.enc_norm(self.embedding(x))


class ExprDecoder(nn.Module):
    def __init__(self, d_model: int, explicit_zero_prob: bool = False,
                 use_batch_labels: bool = False):
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        self.fc = nn.Sequential(
            nn.Linear(d_in, d_model), nn.LeakyReLU(),
            nn.Linear(d_model, d_model), nn.LeakyReLU(),
            nn.Linear(d_model, 1),
        )
        self.explicit_zero_prob = explicit_zero_prob
        if explicit_zero_prob:
            self.zero_logit = nn.Sequential(
                nn.Linear(d_in, d_model), nn.LeakyReLU(),
                nn.Linear(d_model, d_model), nn.LeakyReLU(),
                nn.Linear(d_model, 1),
            )

    def forward(self, x):
        pred_value = self.fc(x).squeeze(-1)
        if not self.explicit_zero_prob:
            return dict(pred=pred_value)
        zero_probs = torch.sigmoid(self.zero_logit(x).squeeze(-1))
        return dict(pred=pred_value, zero_probs=zero_probs)


class ClsDecoder(nn.Module):
    def __init__(self, d_model: int, n_cls: int, nlayers: int = 3):
        super().__init__()
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(nn.ReLU())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)

    def forward(self, x):
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)


class Similarity(nn.Module):
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class BatchLabelEncoder(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx=None):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        return self.enc_norm(self.embedding(x))


class MVCDecoder(nn.Module):
    def __init__(self, d_model, arch_style="inner product",
                 query_activation=nn.Sigmoid, hidden_activation=nn.PReLU,
                 explicit_zero_prob=False, use_batch_labels=False):
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        self.arch_style = arch_style
        self.explicit_zero_prob = explicit_zero_prob
        if arch_style in ["inner product", "inner product, detach"]:
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.W = nn.Linear(d_model, d_in, bias=False)
            if explicit_zero_prob:
                self.W_zero_logit = nn.Linear(d_model, d_in)

    def forward(self, cell_emb, gene_embs):
        if self.arch_style in ["inner product", "inner product, detach"]:
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(2)
            pred_value = torch.bmm(self.W(query_vecs), cell_emb).squeeze(2)
            if not self.explicit_zero_prob:
                return dict(pred=pred_value)
            zero_logits = torch.bmm(self.W_zero_logit(query_vecs), cell_emb).squeeze(2)
            zero_probs = torch.sigmoid(zero_logits)
            return dict(pred=pred_value, zero_probs=zero_probs)


class TransformerModel(nn.Module):
    """Minimal reimplementation of scGPT TransformerModel for cell embedding.

    Only includes the encoder path needed for embed_data.
    Based on bowang-lab/scGPT scgpt/model/model.py.
    """

    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        nlayers_cls: int = 3,
        n_cls: int = 1,
        vocab=None,
        dropout: float = 0.5,
        pad_token: str = "<pad>",
        pad_value: int = -2,
        do_mvc: bool = False,
        do_dab: bool = False,
        use_batch_labels: bool = False,
        num_batch_labels: int = 0,
        domain_spec_batchnorm: bool = False,
        input_emb_style: str = "continuous",
        n_input_bins: int = 0,
        cell_emb_style: str = "cls",
        mvc_decoder_style: str = "inner product",
        ecs_threshold: float = 0.3,
        explicit_zero_prob: bool = False,
        use_fast_transformer: bool = False,
        fast_transformer_backend: str = "flash",
        pre_norm: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.do_dab = do_dab
        self.ecs_threshold = ecs_threshold
        self.use_batch_labels = use_batch_labels
        self.domain_spec_batchnorm = domain_spec_batchnorm
        self.input_emb_style = input_emb_style
        self.cell_emb_style = cell_emb_style
        self.explicit_zero_prob = explicit_zero_prob

        self.encoder = GeneEncoder(ntoken, d_model,
                                   padding_idx=vocab[pad_token] if vocab else 0)

        if input_emb_style == "continuous":
            self.value_encoder = ContinuousValueEncoder(d_model, dropout)
        elif input_emb_style == "category":
            self.value_encoder = CategoryValueEncoder(n_input_bins, d_model,
                                                       padding_idx=pad_value)
        else:
            self.value_encoder = nn.Identity()

        if use_batch_labels:
            self.batch_encoder = BatchLabelEncoder(num_batch_labels, d_model)

        # Standard PyTorch transformer (no flash-attn)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, nlayers)

        self.decoder = ExprDecoder(d_model, explicit_zero_prob=explicit_zero_prob,
                                   use_batch_labels=use_batch_labels)
        self.cls_decoder = ClsDecoder(d_model, n_cls, nlayers=nlayers_cls)

        if do_mvc:
            self.mvc_decoder = MVCDecoder(d_model, arch_style=mvc_decoder_style,
                                          explicit_zero_prob=explicit_zero_prob,
                                          use_batch_labels=use_batch_labels)

        self.sim = Similarity(temp=0.5)
        self.creterion_cce = nn.CrossEntropyLoss()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.embedding.weight.data.uniform_(-initrange, initrange)

    def _encode(self, src, values, src_key_padding_mask, batch_labels=None):
        src = self.encoder(src)
        self.cur_gene_token_embs = src
        values = self.value_encoder(values)

        if self.input_emb_style == "scaling":
            values = values.unsqueeze(2)
            total_embs = src * values
        else:
            total_embs = src + values

        if getattr(self, "bn", None) is not None:
            total_embs = self.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)

        output = self.transformer_encoder(
            total_embs, src_key_padding_mask=src_key_padding_mask
        )
        return output

    def _get_cell_emb_from_layer(self, layer_output, weights=None):
        if self.cell_emb_style == "cls":
            return layer_output[:, 0, :]
        elif self.cell_emb_style == "avg-pool":
            return torch.mean(layer_output, dim=1)
        elif self.cell_emb_style == "w-pool":
            cell_emb = torch.sum(layer_output * weights.unsqueeze(2), dim=1)
            return F.normalize(cell_emb, p=2, dim=1)

    def generate(
        self,
        cell_emb: torch.Tensor,
        src: torch.Tensor,
        values: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        gen_iters: int = 1,
        batch_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate gene expression from cell embedding via scGPT's decoder.

        This re-injects the cell embedding at the [CLS] position (position 0),
        re-runs the full transformer encoder, and applies ExprDecoder to
        predict per-gene expression values.

        Parameters
        ----------
        cell_emb : (B, d_model) cell embeddings (e.g., from DiT output)
        src : (B, seq_len) gene token IDs (with <cls> at position 0)
        values : (B, seq_len) expression values (position 0 = pad_value for <cls>)
        src_key_padding_mask : (B, seq_len) padding mask
        gen_iters : int
            Number of iterative generation refinement steps.
        batch_labels : optional (B,) batch label IDs

        Returns
        -------
        output : (B, seq_len) predicted expression values per gene
        """
        # Encode gene tokens and values
        src_embs = self.encoder(src)                # (B, seq_len, d_model)
        val_embs = self.value_encoder(values)       # (B, seq_len, d_model)

        if self.input_emb_style == "scaling":
            total_embs = src_embs * val_embs.unsqueeze(2)
        else:
            total_embs = src_embs + val_embs

        # Inject cell_emb at position 0 (replacing [CLS] token embedding)
        total_embs[:, 0, :] = cell_emb

        if getattr(self, "bn", None) is not None:
            total_embs = self.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)

        for _ in range(gen_iters):
            output = self.transformer_encoder(
                total_embs, src_key_padding_mask=src_key_padding_mask
            )
            # Update with transformer output for next iteration
            total_embs[:, 0, :] = cell_emb  # Keep cell_emb fixed at position 0

        # Decode: ExprDecoder applied to each gene token output → scalar expression
        mlm_output = self.decoder(output)  # {"pred": (B, seq_len)}
        return mlm_output["pred"]


def load_pretrained(model, pretrained_params, strict=False, verbose=False):
    """Load pretrained params, matching shapes (from scGPT utils)."""
    if isinstance(pretrained_params, dict) and "model_state_dict" in pretrained_params:
        pretrained_params = pretrained_params["model_state_dict"]

    # Handle flash-attn → standard attn key conversion
    pretrained_params = {
        k.replace("Wqkv.", "in_proj_"): v for k, v in pretrained_params.items()
    }

    model_dict = model.state_dict()
    if strict:
        model_dict.update(pretrained_params)
        model.load_state_dict(model_dict)
    else:
        pretrained_params = {
            k: v for k, v in pretrained_params.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        if verbose:
            for k in pretrained_params:
                logger.info(f"Loading: {k}")
        model_dict.update(pretrained_params)
        model.load_state_dict(model_dict)
    return model


# ============================================================================
#  Data Collator (from scGPT data_collator.py, simplified)
# ============================================================================

class CellDataCollator:
    """Collator that pads gene_ids and values, applies binning."""

    def __init__(self, pad_token_id, pad_value, max_length, do_binning=True,
                 n_bins=51):
        self.pad_token_id = pad_token_id
        self.pad_value = pad_value
        self.max_length = max_length
        self.do_binning = do_binning
        self.n_bins = n_bins

    def __call__(self, batch):
        gene_list = [b["genes"] for b in batch]
        value_list = [b["expressions"] for b in batch]

        max_len = min(max(len(g) for g in gene_list), self.max_length)

        padded_genes = []
        padded_values = []

        for genes, values in zip(gene_list, value_list):
            if len(genes) > max_len:
                # Random sample keeping cls at position 0
                idx = np.random.choice(len(genes) - 1, max_len - 1, replace=False) + 1
                idx = np.concatenate([[0], idx])
                genes = genes[idx]
                values = values[idx]

            pad_len = max_len - len(genes)
            if pad_len > 0:
                genes = torch.cat([genes, torch.full((pad_len,), self.pad_token_id, dtype=genes.dtype)])
                values = torch.cat([values, torch.full((pad_len,), self.pad_value, dtype=values.dtype)])

            padded_genes.append(genes)
            padded_values.append(values)

        result = {
            "gene": torch.stack(padded_genes),
            "expr": torch.stack(padded_values),
        }

        # Binning
        if self.do_binning:
            expr = result["expr"]
            # Don't bin special values
            mask = expr != self.pad_value
            if mask.any():
                valid = expr[mask]
                # Rank-based binning
                if valid.numel() > 0:
                    non_zero = valid[valid > 0]
                    if non_zero.numel() > 0:
                        bins = torch.quantile(non_zero.float(),
                                              torch.linspace(0, 1, self.n_bins + 1).to(valid.device))
                        binned = torch.bucketize(expr, bins)
                        binned[~mask] = self.pad_value
                        result["expr"] = binned.long()

        return result


# ============================================================================
#  Main Cell Encoder
# ============================================================================

class ScGPTCellEncoder:
    """Standalone scGPT cell encoder for embedding extraction.

    Parameters
    ----------
    model_dir : str or Path
        Directory containing: best_model.pt, vocab.json, args.json
    device : str or torch.device
    max_length : int
        Max number of genes per cell.
    batch_size : int
        Encoding batch size.
    """

    def __init__(
        self,
        model_dir: Union[str, Path],
        device: Union[str, torch.device] = "cuda",
        max_length: int = 1200,
        batch_size: int = 64,
    ):
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)
        self.max_length = max_length
        self.batch_size = batch_size

        self.model = None
        self.vocab = None
        self.model_configs = None
        # Reference gene set for decoding (populated during encode())
        self._ref_gene_ids = None
        self._ref_gene_names = None

    def _load(self):
        """Load model, vocab, and config."""
        if self.model is not None:
            return

        vocab_file = self.model_dir / "vocab.json"
        config_file = self.model_dir / "args.json"
        model_file = self.model_dir / "best_model.pt"

        for f in [vocab_file, config_file, model_file]:
            if not f.exists():
                raise FileNotFoundError(
                    f"Missing {f.name} in {self.model_dir}. "
                    f"Download scGPT whole-human from: "
                    f"https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y"
                )

        # Load vocab
        self.vocab = SimpleGeneVocab.from_file(vocab_file)
        logger.info(f"Vocabulary: {len(self.vocab)} genes")

        # Load config
        with open(config_file) as f:
            self.model_configs = json.load(f)
        logger.info(f"Model config: d_model={self.model_configs['embsize']}, "
                     f"nlayers={self.model_configs['nlayers']}, "
                     f"nheads={self.model_configs['nheads']}")

        # Special tokens
        pad_token = "<pad>"
        for s in [pad_token, "<cls>", "<eoc>"]:
            if s not in self.vocab:
                self.vocab.append_token(s)
        self.vocab.set_default_index(self.vocab[pad_token])

        # Build model
        self.model = TransformerModel(
            ntoken=len(self.vocab),
            d_model=self.model_configs["embsize"],
            nhead=self.model_configs["nheads"],
            d_hid=self.model_configs["d_hid"],
            nlayers=self.model_configs["nlayers"],
            nlayers_cls=self.model_configs.get("n_layers_cls", 3),
            n_cls=1,
            vocab=self.vocab,
            dropout=self.model_configs.get("dropout", 0.0),
            pad_token=pad_token,
            pad_value=self.model_configs.get("pad_value", -2),
            do_mvc=True,
            do_dab=False,
            use_batch_labels=False,
            domain_spec_batchnorm=False,
            explicit_zero_prob=False,
            use_fast_transformer=False,
        )

        # Load weights
        logger.info(f"Loading weights from {model_file}...")
        state_dict = torch.load(model_file, map_location=self.device)
        load_pretrained(self.model, state_dict, verbose=False)
        self.model.to(self.device)
        self.model.eval()

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"scGPT loaded: {n_params / 1e6:.1f}M parameters")

    @torch.no_grad()
    def encode(
        self,
        adata,
        gene_col: str = "feature_name",
    ) -> np.ndarray:
        """Encode cells to embeddings.

        Parameters
        ----------
        adata : AnnData
            Preprocessed single-cell data. Gene names should be in
            adata.var.index or adata.var[gene_col].
        gene_col : str
            Column in adata.var containing gene names. If not found,
            uses adata.var_names (index).

        Returns
        -------
        embeddings : np.ndarray, shape (n_cells, d_model)
        """
        self._load()

        # Map genes to vocab IDs
        if gene_col in adata.var.columns:
            genes = adata.var[gene_col].tolist()
        else:
            genes = adata.var_names.tolist()

        gene_ids = np.array([
            self.vocab[g] if g in self.vocab else -1 for g in genes
        ], dtype=int)

        # Filter to genes in vocab
        valid_mask = gene_ids >= 0
        n_matched = valid_mask.sum()
        logger.info(f"Matched {n_matched}/{len(genes)} genes in scGPT vocabulary")

        # Case-insensitive retry: mouse gene symbols are Title Case (Xkr4)
        # while scGPT uses UPPERCASE human symbols (XKR4)
        if n_matched < 100:
            logger.info("Low match rate — retrying with UPPERCASE gene names...")
            gene_ids_upper = np.array([
                self.vocab[g.upper()] if g.upper() in self.vocab else -1
                for g in genes
            ], dtype=int)
            valid_mask_upper = gene_ids_upper >= 0
            n_upper = valid_mask_upper.sum()
            logger.info(f"Case-insensitive match: {n_upper}/{len(genes)} genes")
            if n_upper > n_matched:
                gene_ids = gene_ids_upper
                valid_mask = valid_mask_upper
                n_matched = n_upper
                # Update gene names to uppercase for consistency
                genes = [g.upper() for g in genes]

        min_genes = 100  # Need at least 100 matched genes for meaningful embeddings
        if n_matched < min_genes:
            raise ValueError(
                f"Only {n_matched}/{len(genes)} genes matched the scGPT vocabulary "
                f"(minimum {min_genes} required). This dataset likely uses "
                f"non-human gene names or Ensembl IDs."
            )

        adata_filtered = adata[:, valid_mask].copy()
        gene_ids_filtered = gene_ids[valid_mask]

        # Store reference gene set for decoding (use `genes` list which may
        # have been uppercased for case-insensitive matching)
        self._ref_gene_names = np.array(genes)[valid_mask].tolist()
        self._ref_gene_ids = gene_ids_filtered.copy()

        # Get count matrix
        import scipy.sparse as sp
        count_matrix = adata_filtered.X
        if sp.issparse(count_matrix):
            count_matrix = count_matrix.toarray()
        count_matrix = count_matrix.astype(np.float32)

        # Build dataset
        pad_token_id = self.vocab["<pad>"]
        cls_token_id = self.vocab["<cls>"]
        pad_value = self.model_configs.get("pad_value", -2)

        class CellDataset(Dataset):
            def __init__(self, counts, gids):
                self.counts = counts
                self.gids = gids

            def __len__(self):
                return len(self.counts)

            def __getitem__(self, idx):
                row = self.counts[idx]
                nonzero_idx = np.nonzero(row)[0]
                values = row[nonzero_idx]
                gids = self.gids[nonzero_idx]
                # Prepend <cls>
                gids = np.insert(gids, 0, cls_token_id)
                values = np.insert(values, 0, pad_value)
                return {
                    "genes": torch.from_numpy(gids).long(),
                    "expressions": torch.from_numpy(values).float(),
                }

        dataset = CellDataset(count_matrix, gene_ids_filtered)
        collator = CellDataCollator(
            pad_token_id=pad_token_id,
            pad_value=pad_value,
            max_length=self.max_length,
            do_binning=True,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=SequentialSampler(dataset),
            collate_fn=collator,
            drop_last=False,
            num_workers=min(4, self.batch_size),
            pin_memory=True,
        )

        # Encode
        embeddings = np.zeros((len(dataset), self.model_configs["embsize"]),
                              dtype=np.float32)
        count = 0

        for data_dict in tqdm(loader, desc="Encoding cells with scGPT"):
            input_gene_ids = data_dict["gene"].to(self.device)
            input_values = data_dict["expr"].float().to(self.device)
            src_key_padding_mask = input_gene_ids.eq(pad_token_id)

            with torch.amp.autocast("cuda", enabled=True):
                output = self.model._encode(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                )
            # Get [CLS] embedding (position 0)
            emb = output[:, 0, :].cpu().numpy()
            embeddings[count:count + len(emb)] = emb
            count += len(emb)

        logger.info(f"Encoded {count} cells → ({count}, {self.model_configs['embsize']})")
        return embeddings

    @torch.no_grad()
    def decode(
        self,
        cell_embeddings: np.ndarray,
        gene_ids: Optional[np.ndarray] = None,
        gene_names: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
    ) -> dict:
        """Decode cell embeddings back to gene expression via scGPT generate().

        Uses the TransformerModel.generate() method to inject cell embeddings
        at the [CLS] position and reconstruct per-gene expression values.

        Parameters
        ----------
        cell_embeddings : (N, 512) cell embeddings to decode
        gene_ids : (G,) gene token IDs, optional.
            If None, uses reference genes from last encode() call.
        gene_names : list of str, optional.
            Gene names corresponding to gene_ids.
        batch_size : int, optional.
            Decoding batch size.

        Returns
        -------
        result : dict with keys:
            "expression" : (N, G) predicted gene expression matrix
            "gene_names" : list of G gene name strings
        """
        self._load()

        if gene_ids is None:
            if self._ref_gene_ids is None:
                raise ValueError(
                    "No reference gene set available. Call encode() first, "
                    "or provide gene_ids explicitly."
                )
            gene_ids = self._ref_gene_ids
            gene_names = self._ref_gene_names

        if batch_size is None:
            batch_size = self.batch_size

        device = self.device
        pad_token_id = self.vocab["<pad>"]
        cls_token_id = self.vocab["<cls>"]
        pad_value = self.model_configs.get("pad_value", -2)

        N = cell_embeddings.shape[0]
        G = len(gene_ids)

        # Build fixed gene token sequence: [<cls>] + gene_ids
        # Values: pad_value for <cls>, then zeros (will be predicted)
        fixed_genes = np.insert(gene_ids, 0, cls_token_id)   # (G+1,)
        fixed_values = np.full(G + 1, 0.0, dtype=np.float32)
        fixed_values[0] = pad_value  # <cls> position

        src = torch.from_numpy(fixed_genes).long().to(device)           # (G+1,)
        values = torch.from_numpy(fixed_values).float().to(device)      # (G+1,)
        src_key_padding_mask = src.eq(pad_token_id)                     # (G+1,)

        all_preds = []

        for i in tqdm(range(0, N, batch_size), desc="Decoding with scGPT generate()"):
            batch_emb = cell_embeddings[i:i+batch_size]
            if isinstance(batch_emb, np.ndarray):
                batch_emb = torch.from_numpy(batch_emb).float()
            batch_emb = batch_emb.to(device)

            B = batch_emb.shape[0]

            # Expand fixed inputs to batch
            src_batch = src.unsqueeze(0).expand(B, -1)                         # (B, G+1)
            values_batch = values.unsqueeze(0).expand(B, -1)                   # (B, G+1)
            mask_batch = src_key_padding_mask.unsqueeze(0).expand(B, -1)       # (B, G+1)

            with torch.amp.autocast("cuda", enabled=True):
                pred = self.model.generate(
                    cell_emb=batch_emb,
                    src=src_batch,
                    values=values_batch,
                    src_key_padding_mask=mask_batch,
                )
            # pred shape: (B, G+1), skip position 0 (<cls>)
            all_preds.append(pred[:, 1:].float().cpu().numpy())

        expression = np.concatenate(all_preds, axis=0).astype(np.float32)  # (N, G)
        logger.info(f"Decoded {N} cells → expression matrix {expression.shape}")

        return {
            "expression": expression,
            "gene_names": gene_names if gene_names is not None else [f"gene_{i}" for i in range(G)],
        }

    def get_reference_genes(self) -> Optional[dict]:
        """Return the reference gene set from the last encode() call.

        Returns
        -------
        dict with "gene_ids" (np.ndarray) and "gene_names" (list of str),
        or None if no encode() has been called.
        """
        if self._ref_gene_ids is None:
            return None
        return {
            "gene_ids": self._ref_gene_ids,
            "gene_names": self._ref_gene_names,
        }

    @property
    def embed_dim(self) -> int:
        self._load()
        return self.model_configs["embsize"]
