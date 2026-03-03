<template>
  <div id="architecture-root">
    <svg
      :width="svgW"
      :height="svgH"
      :viewBox="`0 0 ${svgW} ${svgH}`"
      xmlns="http://www.w3.org/2000/svg"
      style="font-family: 'Inter', sans-serif; background: #ffffff"
    >
      <defs>
        <!-- Arrowhead marker -->
        <marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto" fill="#555">
          <polygon points="0 0, 10 3.5, 0 7" />
        </marker>
        <marker id="arrow-blue" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto" fill="#3B82F6">
          <polygon points="0 0, 10 3.5, 0 7" />
        </marker>
        <marker id="arrow-green" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto" fill="#10B981">
          <polygon points="0 0, 10 3.5, 0 7" />
        </marker>
        <marker id="arrow-purple" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto" fill="#8B5CF6">
          <polygon points="0 0, 10 3.5, 0 7" />
        </marker>
        <marker id="arrow-orange" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto" fill="#F59E0B">
          <polygon points="0 0, 10 3.5, 0 7" />
        </marker>
        <marker id="arrow-red" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto" fill="#EF4444">
          <polygon points="0 0, 10 3.5, 0 7" />
        </marker>

        <!-- Gradient fills -->
        <linearGradient id="grad-input" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stop-color="#DBEAFE" />
          <stop offset="100%" stop-color="#BFDBFE" />
        </linearGradient>
        <linearGradient id="grad-bert" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stop-color="#E0E7FF" />
          <stop offset="100%" stop-color="#C7D2FE" />
        </linearGradient>
        <linearGradient id="grad-clop" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stop-color="#D1FAE5" />
          <stop offset="100%" stop-color="#A7F3D0" />
        </linearGradient>
        <linearGradient id="grad-dit" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stop-color="#FDE68A" />
          <stop offset="100%" stop-color="#FCD34D" />
        </linearGradient>
        <linearGradient id="grad-decoder" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stop-color="#FECACA" />
          <stop offset="100%" stop-color="#FCA5A5" />
        </linearGradient>
        <linearGradient id="grad-output" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stop-color="#FCE7F3" />
          <stop offset="100%" stop-color="#FBCFE8" />
        </linearGradient>
        <linearGradient id="grad-scgpt-enc" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stop-color="#E0F2FE" />
          <stop offset="100%" stop-color="#BAE6FD" />
        </linearGradient>

        <!-- Filter for subtle shadow -->
        <filter id="shadow" x="-4%" y="-4%" width="108%" height="112%">
          <feDropShadow dx="1" dy="2" stdDeviation="2.5" flood-opacity="0.08" />
        </filter>
        <filter id="shadow-sm" x="-4%" y="-4%" width="108%" height="112%">
          <feDropShadow dx="0.5" dy="1" stdDeviation="1.5" flood-opacity="0.06" />
        </filter>
      </defs>

      <!-- ===================== TITLE ===================== -->
      <text x="660" y="38" text-anchor="middle" font-size="20" font-weight="700" fill="#1E293B" letter-spacing="0.5">
        CLOP-DiT: Text-Guided Single-Cell Generation via Contrastive Alignment and Flow-Matching
      </text>
      <line x1="100" y1="52" x2="1220" y2="52" stroke="#CBD5E1" stroke-width="1" />

      <!-- ===================== PHASE LABELS ===================== -->
      <text x="200" y="80" text-anchor="middle" font-size="11" font-weight="600" fill="#6366F1" letter-spacing="1">
        STAGE I — ENCODING
      </text>
      <text x="540" y="80" text-anchor="middle" font-size="11" font-weight="600" fill="#059669" letter-spacing="1">
        STAGE II — ALIGNMENT
      </text>
      <text x="830" y="80" text-anchor="middle" font-size="11" font-weight="600" fill="#D97706" letter-spacing="1">
        STAGE III — GENERATION
      </text>
      <text x="1120" y="80" text-anchor="middle" font-size="11" font-weight="600" fill="#DC2626" letter-spacing="1">
        STAGE IV — DECODING
      </text>

      <!-- ===================== INPUT TEXT BLOCK ===================== -->
      <g transform="translate(40, 115)" filter="url(#shadow)">
        <rect width="130" height="130" rx="10" fill="url(#grad-input)" stroke="#93C5FD" stroke-width="1.5" />
        <text x="65" y="22" text-anchor="middle" font-size="11" font-weight="700" fill="#1E40AF">Text Prompt</text>
        <line x1="12" y1="30" x2="118" y2="30" stroke="#93C5FD" stroke-width="0.8" />
        <!-- Sample text lines -->
        <text x="65" y="48" text-anchor="middle" font-size="8" fill="#3B82F6" font-style="italic">"CD8+ cytotoxic T cell</text>
        <text x="65" y="59" text-anchor="middle" font-size="8" fill="#3B82F6" font-style="italic">from non-small cell</text>
        <text x="65" y="70" text-anchor="middle" font-size="8" fill="#3B82F6" font-style="italic">lung carcinoma with</text>
        <text x="65" y="81" text-anchor="middle" font-size="8" fill="#3B82F6" font-style="italic">high CD8A, GZMB, PRF1"</text>
        <line x1="12" y1="92" x2="118" y2="92" stroke="#93C5FD" stroke-width="0.5" stroke-dasharray="3,2" />
        <text x="65" y="108" text-anchor="middle" font-size="8" fill="#64748B">Cell type + tissue +</text>
        <text x="65" y="119" text-anchor="middle" font-size="8" fill="#64748B">marker genes</text>
      </g>

      <!-- Arrow: Input → BiomedBERT -->
      <line x1="172" y1="180" x2="205" y2="180" stroke="#555" stroke-width="1.5" marker-end="url(#arrow)" />

      <!-- ===================== BiomedBERT BLOCK ===================== -->
      <g transform="translate(208, 105)" filter="url(#shadow)">
        <rect width="150" height="150" rx="10" fill="url(#grad-bert)" stroke="#A5B4FC" stroke-width="1.5" />
        <text x="75" y="22" text-anchor="middle" font-size="11" font-weight="700" fill="#4338CA">BiomedBERT-large</text>
        <line x1="12" y1="30" x2="138" y2="30" stroke="#A5B4FC" stroke-width="0.8" />
        <!-- Transformer layers stacked -->
        <rect x="15" y="38" width="120" height="18" rx="4" fill="#EEF2FF" stroke="#A5B4FC" stroke-width="0.8" />
        <text x="75" y="51" text-anchor="middle" font-size="7.5" fill="#4338CA">Transformer Layer ×12</text>
        <rect x="15" y="62" width="120" height="18" rx="4" fill="#EEF2FF" stroke="#A5B4FC" stroke-width="0.8" />
        <text x="75" y="75" text-anchor="middle" font-size="7.5" fill="#4338CA">Self-Attention (16 heads)</text>
        <rect x="15" y="86" width="120" height="18" rx="4" fill="#EEF2FF" stroke="#A5B4FC" stroke-width="0.8" />
        <text x="75" y="99" text-anchor="middle" font-size="7.5" fill="#4338CA">[CLS] Pooling</text>
        <line x1="12" y1="112" x2="138" y2="112" stroke="#A5B4FC" stroke-width="0.5" stroke-dasharray="3,2" />
        <text x="75" y="127" text-anchor="middle" font-size="9" font-weight="600" fill="#4338CA">Output: ℝ</text>
        <text x="104" y="124" font-size="7" font-weight="600" fill="#4338CA">1024</text>
        <text x="75" y="143" text-anchor="middle" font-size="7.5" fill="#64748B">Frozen Pretrained</text>
      </g>

      <!-- Arrow: BiomedBERT → Whitening -->
      <line x1="360" y1="180" x2="393" y2="180" stroke="#555" stroke-width="1.5" marker-end="url(#arrow)" />

      <!-- ===================== PCA WHITENING ===================== -->
      <g transform="translate(396, 152)" filter="url(#shadow-sm)">
        <rect width="90" height="56" rx="8" fill="#F5F3FF" stroke="#C4B5FD" stroke-width="1.2" />
        <text x="45" y="18" text-anchor="middle" font-size="9" font-weight="600" fill="#7C3AED">PCA Whitening</text>
        <text x="45" y="33" text-anchor="middle" font-size="7.5" fill="#7C3AED" font-style="italic">x̂ = (x − μ) W⊤</text>
        <text x="45" y="47" text-anchor="middle" font-size="7" fill="#64748B">Decorrelation</text>
      </g>

      <!-- Arrow: Whitening → CLOP Text -->
      <line x1="488" y1="180" x2="508" y2="155" stroke="#10B981" stroke-width="1.5" marker-end="url(#arrow-green)" />

      <!-- ===================== CLOP ALIGNMENT MODULE ===================== -->
      <g transform="translate(500, 95)" filter="url(#shadow)">
        <rect width="170" height="250" rx="12" fill="url(#grad-clop)" stroke="#6EE7B7" stroke-width="1.8" />
        <text x="85" y="22" text-anchor="middle" font-size="12" font-weight="700" fill="#065F46">CLOP Aligner</text>
        <line x1="12" y1="30" x2="158" y2="30" stroke="#6EE7B7" stroke-width="0.8" />

        <!-- Text Projector sub-block -->
        <rect x="12" y="38" width="146" height="55" rx="6" fill="#ECFDF5" stroke="#6EE7B7" stroke-width="1" />
        <text x="85" y="53" text-anchor="middle" font-size="9" font-weight="600" fill="#047857">Text Projector</text>
        <text x="85" y="66" text-anchor="middle" font-size="7.5" fill="#047857">MLP (3 layers) + BN</text>
        <text x="85" y="79" text-anchor="middle" font-size="7.5" fill="#047857">1024 → 512 → 256</text>
        <text x="85" y="88" text-anchor="middle" font-size="6" fill="#64748B">+ L2 Normalize</text>

        <!-- Cell Projector sub-block -->
        <rect x="12" y="100" width="146" height="55" rx="6" fill="#ECFDF5" stroke="#6EE7B7" stroke-width="1" />
        <text x="85" y="115" text-anchor="middle" font-size="9" font-weight="600" fill="#047857">Cell Projector</text>
        <text x="85" y="128" text-anchor="middle" font-size="7.5" fill="#047857">MLP (3 layers) + BN</text>
        <text x="85" y="141" text-anchor="middle" font-size="7.5" fill="#047857">512 → 384 → 256</text>
        <text x="85" y="150" text-anchor="middle" font-size="6" fill="#64748B">+ L2 Normalize</text>

        <!-- Loss block -->
        <rect x="12" y="163" width="146" height="50" rx="6" fill="#D1FAE5" stroke="#34D399" stroke-width="1" />
        <text x="85" y="178" text-anchor="middle" font-size="9" font-weight="600" fill="#065F46">InfoNCE Loss</text>
        <text x="85" y="193" text-anchor="middle" font-size="7" fill="#065F46" font-style="italic">ℒ = −log(exp(s⁺/τ) / Σ exp(sᵢ/τ))</text>
        <text x="85" y="207" text-anchor="middle" font-size="7" fill="#065F46">τ learnable (init 0.07)</text>

        <!-- Output dim -->
        <text x="85" y="230" text-anchor="middle" font-size="9" font-weight="600" fill="#065F46">Shared Space: ℝ²⁵⁶</text>
        <text x="85" y="243" text-anchor="middle" font-size="7" fill="#64748B">Label smoothing: 0.1</text>
      </g>

      <!-- ===================== scGPT CELL ENCODER (below, feeding into CLOP) ===================== -->
      <g transform="translate(210, 315)" filter="url(#shadow)">
        <rect width="150" height="120" rx="10" fill="url(#grad-scgpt-enc)" stroke="#7DD3FC" stroke-width="1.5" />
        <text x="75" y="20" text-anchor="middle" font-size="11" font-weight="700" fill="#0369A1">scGPT Encoder</text>
        <line x1="12" y1="28" x2="138" y2="28" stroke="#7DD3FC" stroke-width="0.8" />
        <rect x="12" y="35" width="126" height="16" rx="4" fill="#F0F9FF" stroke="#7DD3FC" stroke-width="0.7" />
        <text x="75" y="47" text-anchor="middle" font-size="7.5" fill="#0369A1">Gene Tokenizer (1890)</text>
        <rect x="12" y="55" width="126" height="16" rx="4" fill="#F0F9FF" stroke="#7DD3FC" stroke-width="0.7" />
        <text x="75" y="67" text-anchor="middle" font-size="7.5" fill="#0369A1">Transformer (Pan-cancer)</text>
        <rect x="12" y="75" width="126" height="16" rx="4" fill="#F0F9FF" stroke="#7DD3FC" stroke-width="0.7" />
        <text x="75" y="87" text-anchor="middle" font-size="7.5" fill="#0369A1">[CLS] → Cell Embedding</text>
        <text x="75" y="107" text-anchor="middle" font-size="9" font-weight="600" fill="#0369A1">Output: ℝ⁵¹²</text>
        <text x="75" y="117" text-anchor="middle" font-size="7" fill="#64748B">Frozen Pretrained</text>
      </g>

      <!-- Arrow: scGPT Encoder → CLOP Cell Projector (going right + up) -->
      <path d="M 362 375 L 430 375 Q 490 375 490 330 L 490 225 L 498 225"
            fill="none" stroke="#0EA5E9" stroke-width="1.5" marker-end="url(#arrow-blue)" />
      <text x="420" y="368" font-size="7" fill="#0369A1" font-weight="500">ℝ⁵¹²</text>

      <!-- Real cell data input to scGPT -->
      <g transform="translate(40, 345)" filter="url(#shadow-sm)">
        <rect width="130" height="60" rx="8" fill="#F0F9FF" stroke="#7DD3FC" stroke-width="1.2" />
        <text x="65" y="18" text-anchor="middle" font-size="9" font-weight="600" fill="#0369A1">Real Cell Data</text>
        <line x1="12" y1="25" x2="118" y2="25" stroke="#7DD3FC" stroke-width="0.5" stroke-dasharray="3,2" />
        <text x="65" y="38" text-anchor="middle" font-size="7.5" fill="#0369A1">Gene Expression Matrix</text>
        <text x="65" y="50" text-anchor="middle" font-size="7.5" fill="#64748B">190,826 cells × 1890 genes</text>
      </g>

      <!-- Arrow: Real Cell → scGPT -->
      <line x1="172" y1="375" x2="208" y2="375" stroke="#0EA5E9" stroke-width="1.5" marker-end="url(#arrow-blue)" />

      <!-- Arrow: CLOP → DiT (text branch output) -->
      <line x1="672" y1="175" x2="710" y2="175" stroke="#D97706" stroke-width="1.8" marker-end="url(#arrow-orange)" />
      <text x="690" y="167" font-size="7.5" fill="#D97706" font-weight="600">c ∈ ℝ²⁵⁶</text>

      <!-- ===================== DiT FLOW MATCHING ===================== -->
      <g transform="translate(713, 95)" filter="url(#shadow)">
        <rect width="195" height="250" rx="12" fill="url(#grad-dit)" stroke="#FBBF24" stroke-width="1.8" />
        <text x="97" y="22" text-anchor="middle" font-size="12" font-weight="700" fill="#92400E">DiT Flow Matching</text>
        <line x1="12" y1="30" x2="183" y2="30" stroke="#FBBF24" stroke-width="0.8" />

        <!-- Noise input -->
        <rect x="12" y="37" width="171" height="30" rx="5" fill="#FFFBEB" stroke="#FBBF24" stroke-width="0.8" />
        <text x="97" y="50" text-anchor="middle" font-size="8" font-weight="600" fill="#92400E">z₀ ~ 𝒩(0, I) → z_t = (1−t)z₀ + tz₁</text>
        <text x="97" y="62" text-anchor="middle" font-size="7" fill="#92400E">Logit-Normal t sampling</text>

        <!-- Timestep embedder -->
        <rect x="12" y="73" width="80" height="26" rx="5" fill="#FEF3C7" stroke="#F59E0B" stroke-width="0.8" />
        <text x="52" y="86" text-anchor="middle" font-size="7.5" font-weight="600" fill="#92400E">t → Sinusoidal</text>
        <text x="52" y="95" text-anchor="middle" font-size="6.5" fill="#92400E">→ MLP → t_emb</text>

        <!-- Condition embedder -->
        <rect x="100" y="73" width="83" height="26" rx="5" fill="#FEF3C7" stroke="#F59E0B" stroke-width="0.8" />
        <text x="141" y="86" text-anchor="middle" font-size="7.5" font-weight="600" fill="#92400E">c → MLP</text>
        <text x="141" y="95" text-anchor="middle" font-size="6.5" fill="#92400E">CFG drop p=0.15</text>

        <!-- Combined conditioning -->
        <text x="97" y="112" text-anchor="middle" font-size="7.5" fill="#92400E" font-weight="500">cond = t_emb + c_emb</text>

        <!-- DiT blocks -->
        <rect x="12" y="118" width="171" height="60" rx="6" fill="#FFFBEB" stroke="#F59E0B" stroke-width="1" />
        <text x="97" y="133" text-anchor="middle" font-size="9" font-weight="700" fill="#92400E">DiT Block × 8</text>
        <text x="97" y="147" text-anchor="middle" font-size="7.5" fill="#92400E">AdaLN-Zero Modulation</text>
        <text x="97" y="159" text-anchor="middle" font-size="7.5" fill="#92400E">Self-Attn (6 heads) → FFN (4× MLP)</text>
        <text x="97" y="171" text-anchor="middle" font-size="7" fill="#92400E">α-gate: zero-init → identity start</text>

        <!-- Velocity output -->
        <rect x="12" y="184" width="171" height="28" rx="5" fill="#FEF3C7" stroke="#F59E0B" stroke-width="0.8" />
        <text x="97" y="198" text-anchor="middle" font-size="8" font-weight="600" fill="#92400E">v̂(z_t, t, c) → velocity prediction</text>
        <text x="97" y="208" text-anchor="middle" font-size="7" fill="#92400E">ℒ = MSE(v̂, z₁ − z₀)</text>

        <!-- ODE Integration -->
        <rect x="30" y="218" width="135" height="25" rx="5" fill="#FDE68A" stroke="#D97706" stroke-width="1" />
        <text x="97" y="234" text-anchor="middle" font-size="8" font-weight="600" fill="#78350F">ODE Integration (20 steps)</text>
      </g>

      <!-- Arrow: DiT → Decoder -->
      <line x1="910" y1="220" x2="945" y2="220" stroke="#EF4444" stroke-width="1.8" marker-end="url(#arrow-red)" />
      <text x="928" y="212" font-size="7.5" fill="#EF4444" font-weight="600">ẑ₁ ∈ ℝ⁵¹²</text>

      <!-- ===================== scGPT DECODER ===================== -->
      <g transform="translate(948, 120)" filter="url(#shadow)">
        <rect width="150" height="200" rx="10" fill="url(#grad-decoder)" stroke="#F87171" stroke-width="1.5" />
        <text x="75" y="22" text-anchor="middle" font-size="11" font-weight="700" fill="#991B1B">scGPT Decoder</text>
        <line x1="12" y1="30" x2="138" y2="30" stroke="#F87171" stroke-width="0.8" />

        <rect x="12" y="38" width="126" height="20" rx="4" fill="#FFF1F2" stroke="#F87171" stroke-width="0.7" />
        <text x="75" y="52" text-anchor="middle" font-size="7.5" fill="#991B1B">Inject ẑ₁ at [CLS]</text>

        <rect x="12" y="64" width="126" height="20" rx="4" fill="#FFF1F2" stroke="#F87171" stroke-width="0.7" />
        <text x="75" y="78" text-anchor="middle" font-size="7.5" fill="#991B1B">Gene Token Encoding</text>

        <rect x="12" y="90" width="126" height="20" rx="4" fill="#FFF1F2" stroke="#F87171" stroke-width="0.7" />
        <text x="75" y="104" text-anchor="middle" font-size="7.5" fill="#991B1B">Transformer Decoder</text>

        <rect x="12" y="116" width="126" height="20" rx="4" fill="#FFF1F2" stroke="#F87171" stroke-width="0.7" />
        <text x="75" y="130" text-anchor="middle" font-size="7.5" fill="#991B1B">ExprDecoder Head</text>

        <line x1="12" y1="145" x2="138" y2="145" stroke="#F87171" stroke-width="0.5" stroke-dasharray="3,2" />
        <text x="75" y="162" text-anchor="middle" font-size="9" font-weight="600" fill="#991B1B">Output: ℝ¹⁸⁹⁰</text>
        <text x="75" y="177" text-anchor="middle" font-size="7.5" fill="#991B1B">Gene Expression</text>
        <text x="75" y="192" text-anchor="middle" font-size="7" fill="#64748B">Pretrained (generate())</text>
      </g>

      <!-- Arrow: Decoder → Output -->
      <line x1="1100" y1="220" x2="1130" y2="220" stroke="#555" stroke-width="1.5" marker-end="url(#arrow)" />

      <!-- ===================== OUTPUT ===================== -->
      <g transform="translate(1133, 155)" filter="url(#shadow)">
        <rect width="140" height="130" rx="10" fill="url(#grad-output)" stroke="#F9A8D4" stroke-width="1.5" />
        <text x="70" y="20" text-anchor="middle" font-size="10" font-weight="700" fill="#9D174D">Generated Cell</text>
        <line x1="12" y1="28" x2="128" y2="28" stroke="#F9A8D4" stroke-width="0.8" />
        <!-- Simulated heatmap bars -->
        <g transform="translate(14, 35)">
          <rect x="0" y="0" width="112" height="6" rx="2" fill="#FDE68A" opacity="0.4" />
          <rect x="0" y="0" width="42" height="6" rx="2" fill="#EF4444" opacity="0.7" />
          <text x="116" y="6" font-size="5.5" fill="#64748B">CD8A ↑</text>
          <rect x="0" y="10" width="112" height="6" rx="2" fill="#FDE68A" opacity="0.4" />
          <rect x="0" y="10" width="68" height="6" rx="2" fill="#F59E0B" opacity="0.7" />
          <text x="116" y="16" font-size="5.5" fill="#64748B">GZMB ↑</text>
          <rect x="0" y="20" width="112" height="6" rx="2" fill="#FDE68A" opacity="0.4" />
          <rect x="0" y="20" width="55" height="6" rx="2" fill="#F97316" opacity="0.7" />
          <text x="116" y="26" font-size="5.5" fill="#64748B">PRF1 ↑</text>
          <rect x="0" y="30" width="112" height="6" rx="2" fill="#E0F2FE" opacity="0.4" />
          <rect x="0" y="30" width="12" height="6" rx="2" fill="#3B82F6" opacity="0.5" />
          <text x="116" y="36" font-size="5.5" fill="#64748B">MS4A1 ↓</text>
          <rect x="0" y="40" width="112" height="6" rx="2" fill="#E0F2FE" opacity="0.4" />
          <rect x="0" y="40" width="8" height="6" rx="2" fill="#6366F1" opacity="0.5" />
          <text x="116" y="46" font-size="5.5" fill="#64748B">CD68 ↓</text>
        </g>
        <line x1="12" y1="94" x2="128" y2="94" stroke="#F9A8D4" stroke-width="0.5" stroke-dasharray="3,2" />
        <text x="70" y="108" text-anchor="middle" font-size="8" fill="#9D174D" font-weight="600">1890 gene values</text>
        <text x="70" y="120" text-anchor="middle" font-size="7" fill="#64748B">Biologically valid</text>
      </g>

      <!-- ===================== TRAINING vs INFERENCE ANNOTATION ===================== -->
      <!-- Training path bracket (top) -->
      <rect x="40" y="455" width="1240" height="52" rx="8" fill="#F8FAFC" stroke="#E2E8F0" stroke-width="1" />

      <!-- Training info -->
      <text x="60" y="473" font-size="9" font-weight="700" fill="#475569">Training:</text>
      <text x="120" y="473" font-size="8" fill="#64748B">Stage I — Freeze BiomedBERT + scGPT, train CLOP projectors (200 epochs, bs=256, lr=3×10⁻⁴)</text>
      <text x="120" y="488" font-size="8" fill="#64748B">Stage II — Freeze CLOP + scGPT, train DiT (200 epochs, bs=1024, lr=2×10⁻⁴, EMA decay=0.9999)</text>

      <!-- Inference info -->
      <text x="680" y="473" font-size="9" font-weight="700" fill="#475569">Inference:</text>
      <text x="740" y="473" font-size="8" fill="#64748B">Text → BiomedBERT → CLOP TextProj → DiT ODE (20 steps, cfg=3.0) → scGPT decode → Gene Expr</text>
      <text x="740" y="488" font-size="8" fill="#64748B">End-to-end: ~0.3s per cell type (batch of 200 cells), controllable via natural language</text>

      <!-- ===================== KEY INNOVATIONS BOX ===================== -->
      <g transform="translate(40, 515)">
        <rect width="400" height="68" rx="8" fill="#FEFCE8" stroke="#FDE047" stroke-width="1" />
        <text x="15" y="16" font-size="9" font-weight="700" fill="#854D0E">Key Innovations</text>
        <line x1="15" y1="21" x2="385" y2="21" stroke="#FDE047" stroke-width="0.5" />
        <circle cx="15" cy="32" r="2.5" fill="#D97706" />
        <text x="24" y="35" font-size="7" fill="#78350F">Flow matching (OT path) for faster, higher-quality generation</text>
        <circle cx="15" cy="43" r="2.5" fill="#D97706" />
        <text x="24" y="46" font-size="7" fill="#78350F">AdaLN-Zero: zero-initialized gates for stable identity-start training</text>
        <circle cx="15" cy="54" r="2.5" fill="#059669" />
        <text x="24" y="57" font-size="7" fill="#065F46">CLIP-style contrastive alignment bridges language ↔ single-cell omics</text>
        <circle cx="15" cy="65" r="2.5" fill="#059669" />
        <text x="24" y="68" font-size="7" fill="#065F46">Classifier-free guidance (CFG) enables controllable generation strength</text>
      </g>

      <!-- Data stats box -->
      <g transform="translate(460, 515)">
        <rect width="400" height="68" rx="8" fill="#EFF6FF" stroke="#93C5FD" stroke-width="1" />
        <text x="15" y="16" font-size="9" font-weight="700" fill="#1E40AF">Data &amp; Scale</text>
        <line x1="15" y1="21" x2="385" y2="21" stroke="#93C5FD" stroke-width="0.5" />
        <circle cx="15" cy="32" r="2.5" fill="#3B82F6" />
        <text x="24" y="35" font-size="7" fill="#1E3A5F">190,826 cells from 69 cancer datasets (NSCLC, pan-cancer)</text>
        <circle cx="15" cy="43" r="2.5" fill="#3B82F6" />
        <text x="24" y="46" font-size="7" fill="#1E3A5F">6 cell types: CD8⁺ T, Macrophage, Epithelial, Fibroblast, NK, B</text>
        <circle cx="15" cy="54" r="2.5" fill="#8B5CF6" />
        <text x="24" y="57" font-size="7" fill="#4C1D95">1890 genes per cell, scGPT pan-cancer pretrained weights</text>
        <circle cx="15" cy="65" r="2.5" fill="#8B5CF6" />
        <text x="24" y="68" font-size="7" fill="#4C1D95">Group-level train/val split (63 train / 6 val samples)</text>
      </g>

      <!-- Model specs box -->
      <g transform="translate(880, 515)">
        <rect width="400" height="68" rx="8" fill="#F0FDF4" stroke="#86EFAC" stroke-width="1" />
        <text x="15" y="16" font-size="9" font-weight="700" fill="#166534">Model Specifications</text>
        <line x1="15" y1="21" x2="385" y2="21" stroke="#86EFAC" stroke-width="0.5" />
        <circle cx="15" cy="32" r="2.5" fill="#22C55E" />
        <text x="24" y="35" font-size="7" fill="#14532D">DiT: hidden 384-d, 8 transformer blocks, 6 attention heads, 16 tokens</text>
        <circle cx="15" cy="43" r="2.5" fill="#22C55E" />
        <text x="24" y="46" font-size="7" fill="#14532D">CLOP: projection 256-d, learnable τ (init 0.07), label smoothing 0.1</text>
        <circle cx="15" cy="54" r="2.5" fill="#22C55E" />
        <text x="24" y="57" font-size="7" fill="#14532D">EMA decay 0.9999, CFG scale 3.0, ODE integration 20 steps</text>
        <circle cx="15" cy="65" r="2.5" fill="#22C55E" />
        <text x="24" y="68" font-size="7" fill="#14532D">Frozen encoders: BiomedBERT-large (1024-d), scGPT pan-cancer (512-d)</text>
      </g>

      <!-- ===================== DASHED TRAINING-ONLY PATH ===================== -->
      <!-- Dashed line showing "training only" path from real cells through scGPT to CLOP -->
      <text x="285" y="308" text-anchor="middle" font-size="7" fill="#0369A1" font-style="italic">Training only</text>
      <line x1="285" y1="274" x2="285" y2="313" stroke="#0EA5E9" stroke-width="1" stroke-dasharray="4,3" />

    </svg>
  </div>
</template>

<script setup>
const svgW = 1320
const svgH = 600
</script>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}
body {
  background: #ffffff;
  display: flex;
  justify-content: center;
  align-items: flex-start;
  min-height: 100vh;
  padding: 10px;
}
#architecture-root {
  display: inline-block;
}
</style>
