# constants.py — Centralized default constants for CLOP-DiT
"""
Single source of truth for numeric and string defaults used across the codebase.

These constants serve as fallback defaults when YAML configs don't specify a value.
They eliminate scattered magic numbers and ensure consistency across modules.

Usage:
    from src.utils.constants import RANDOM_SEED, CFG_SCALE, LATENT_DIM
"""

# ════════════════════════════════════════════════════════════════════════
#  Reproducibility
# ════════════════════════════════════════════════════════════════════════
RANDOM_SEED = 42

# ════════════════════════════════════════════════════════════════════════
#  Model Dimensions
# ════════════════════════════════════════════════════════════════════════
LATENT_DIM = 512          # Cell embedding / scGPT output dimension
TEXT_DIM_LARGE = 1024     # BiomedBERT-large output dimension
TEXT_DIM_BASE = 768       # BiomedBERT-base output dimension
PROJ_DIM = 512            # CLOP shared projection space dimension

# ════════════════════════════════════════════════════════════════════════
#  DiT Architecture Defaults
# ════════════════════════════════════════════════════════════════════════
DIT_HIDDEN_DIM = 384      # DiT transformer hidden dimension
DIT_NUM_BLOCKS = 8        # Number of DiT transformer blocks
DIT_NUM_HEADS = 6         # Number of attention heads
DIT_MLP_RATIO = 4.0       # FFN expansion ratio
DIT_NUM_TOKENS = 16       # Pseudo-tokens for latent tokenization

# ════════════════════════════════════════════════════════════════════════
#  Diffusion / Flow Matching
# ════════════════════════════════════════════════════════════════════════
CFG_SCALE = 3.0           # Classifier-free guidance scale (inference default)
INFERENCE_STEPS = 20      # ODE solver steps for generation
COND_DROP_PROB = 0.1      # CFG dropout probability during training

# ════════════════════════════════════════════════════════════════════════
#  CLOP Aligner Defaults
# ════════════════════════════════════════════════════════════════════════
CLOP_NUM_LAYERS = 3       # Projection head layers (text & cell)
CLOP_DROPOUT = 0.1        # Projection head dropout
CLOP_TEMPERATURE = 0.07   # InfoNCE temperature init
CLOP_MIN_TEMPERATURE = 0.01
CLOP_MAX_TEMPERATURE = 0.5
CLOP_LABEL_SMOOTHING = 0.1
CLOP_COHESION_WEIGHT = 0.1
CLOP_SEPARATION_THRESHOLD = 0.3
CLOP_WHITENING_EPS = 1e-4
CLOP_SOFT_LABEL_ALPHA = 2.0
CLOP_SOFT_LABEL_BIAS = 5.0

# SigLIP / PrototypeSigLIP defaults
SIGLIP_INIT_TEMPERATURE = 10.0
SIGLIP_INIT_BIAS = -10.0

# ════════════════════════════════════════════════════════════════════════
#  Training Defaults
# ════════════════════════════════════════════════════════════════════════
DEFAULT_LR = 1e-4         # Learning rate (DiT / general)
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_EPOCHS = 10
DEFAULT_NUM_EPOCHS = 200
DEFAULT_EMA_DECAY = 0.9999
DEFAULT_GRAD_CLIP = 1.0
ADAMW_BETAS = (0.9, 0.999)

# ════════════════════════════════════════════════════════════════════════
#  Dropout (shared across architectures)
# ════════════════════════════════════════════════════════════════════════
DEFAULT_ATTN_DROP = 0.0
DEFAULT_PROJ_DROP = 0.1

# ════════════════════════════════════════════════════════════════════════
#  Data Pipeline
# ════════════════════════════════════════════════════════════════════════
DEFAULT_BATCH_SIZE = 512
DEFAULT_MAX_SEQ_LENGTH = 512  # Tokenizer / text processing max length
DEFAULT_NUM_WORKERS = 4
DEFAULT_VAL_SPLIT = 0.1

# ════════════════════════════════════════════════════════════════════════
#  Evaluation
# ════════════════════════════════════════════════════════════════════════
EVAL_BATCH_SIZE = 512         # Batch size for evaluation projection
EVAL_NUM_SAMPLES = 500        # Default samples for generation evaluation
EVAL_TEST_FRACTION = 0.2      # Train/test split for downstream classifiers
CLASSIFIER_MAX_ITER = 2000    # LogisticRegression convergence iterations
DISC_CLASSIFIER_MAX_ITER = 1000  # Discriminator classifier iterations
UMAP_N_NEIGHBORS = 15        # Default UMAP/scanpy neighbor count
UMAP_MIN_DIST = 0.1          # Default UMAP min_dist

# ════════════════════════════════════════════════════════════════════════
#  Model Identifiers (HuggingFace)
# ════════════════════════════════════════════════════════════════════════
BIOMEDBERT_LARGE = "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract"
BIOMEDBERT_BASE = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"

# ════════════════════════════════════════════════════════════════════════
#  Numerical Stability
# ════════════════════════════════════════════════════════════════════════
NORM_EPS = 1e-8               # Epsilon for norm division
LAYERNORM_EPS = 1e-6          # LayerNorm epsilon
