# subcluster_annotation.py — Sub-cluster text description generation for CLOP-DiT
"""Leiden clustering + marker-based annotation + context-aware text generation."""
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scanpy as sc

logger = logging.getLogger(__name__)


CELL_TYPE_SIGNATURES = {
    # ── Immune: Lymphoid ──
    "CD8+ T cells": {
        "markers": ["CD8A", "CD8B", "CD3E", "CD3D", "GZMB", "PRF1", "IFNG", "NKG7", "GZMA", "GZMK", "EOMES", "TBX21", "KLRG1"],
        "description": "CD8+ cytotoxic T lymphocytes",
    },
    "CD4+ T cells": {
        "markers": ["CD4", "IL7R", "CD3E", "CD3D", "TCF7", "LEF1", "CCR7", "SELL", "CD28", "CD40LG", "ICOS"],
        "description": "CD4+ helper T lymphocytes",
    },
    "Regulatory T cells": {
        "markers": ["FOXP3", "IL2RA", "CTLA4", "IKZF2", "TNFRSF18", "CD4", "TIGIT", "LAYN"],
        "description": "CD4+FOXP3+ regulatory T cells (Tregs)",
    },
    "Gamma-delta T cells": {
        "markers": ["TRGC1", "TRGC2", "TRDC", "TRDV2", "TRGV9", "CD3E", "CD3D", "GNLY", "NKG7"],
        "description": "gamma-delta T lymphocytes",
    },
    "Exhausted T cells": {
        "markers": ["PDCD1", "LAG3", "HAVCR2", "TIGIT", "CTLA4", "TOX", "ENTPD1", "CD8A"],
        "description": "exhausted T cells with chronic activation markers",
    },
    "NK cells": {
        "markers": ["NKG7", "KLRD1", "GNLY", "NCR1", "NCAM1", "KLRF1", "FCGR3A", "CD160", "GZMB", "KLRB1", "KLRC1"],
        "description": "natural killer cells",
    },
    "ILC cells": {
        "markers": ["IL7R", "RORC", "IL22", "IL23R", "KIT", "GATA3", "IL1RL1", "KLRB1"],
        "description": "innate lymphoid cells",
    },
    "B cells": {
        "markers": ["CD79A", "CD79B", "MS4A1", "CD19", "PAX5", "BANK1", "BLK", "IGHM", "IGHD", "CD22", "TCL1A"],
        "description": "B lymphocytes",
    },
    "Plasma cells": {
        "markers": ["JCHAIN", "MZB1", "SDC1", "IGHG1", "IGHG2", "IGHA1", "XBP1", "PRDM1", "IRF4"],
        "description": "antibody-secreting plasma cells",
    },

    # ── Immune: Myeloid ──
    "Macrophages": {
        "markers": ["CD68", "CD163", "CSF1R", "MRC1", "MSR1", "MARCO", "C1QA", "C1QB", "APOE", "FCGR1A"],
        "description": "tissue-resident macrophages",
    },
    "Alveolar macrophages": {
        "markers": ["MARCO", "FABP4", "MCEMP1", "RBP4", "PPARG", "SIGLEC1", "CD68", "MSR1"],
        "description": "alveolar macrophages",
    },
    "Kupffer cells": {
        "markers": ["CD163", "MARCO", "CD5L", "TIMD4", "CLEC4F", "VSIG4", "C1QA", "CD68"],
        "description": "liver Kupffer cells (tissue-resident macrophages)",
    },
    "Monocytes": {
        "markers": ["CD14", "FCGR3A", "LYZ", "S100A8", "S100A9", "VCAN", "FCN1", "CST3", "MNDA"],
        "description": "circulating monocytes",
    },
    "Dendritic cells": {
        "markers": ["FCER1A", "CD1C", "CLEC10A", "ITGAX", "HLA-DRA", "HLA-DQA1", "IRF8", "BATF3"],
        "description": "dendritic cells",
    },
    "Plasmacytoid DCs": {
        "markers": ["CLEC4C", "IL3RA", "NRP1", "TCF4", "IRF7", "LILRA4", "GZMB", "JCHAIN"],
        "description": "plasmacytoid dendritic cells",
    },
    "Langerhans cells": {
        "markers": ["CD207", "CD1A", "EPCAM", "FCER1A", "LANGERIN", "CD1C"],
        "description": "Langerhans cells (skin-resident dendritic cells)",
    },
    "Neutrophils": {
        "markers": ["S100A8", "S100A9", "CSF3R", "FCGR3B", "CXCR2", "MMP9", "ELANE", "MPO"],
        "description": "neutrophils",
    },
    "Mast cells": {
        "markers": ["KIT", "CPA3", "TPSAB1", "TPSB2", "HPGDS", "MS4A2", "HDC"],
        "description": "mast cells",
    },
    "Megakaryocytes": {
        "markers": ["ITGA2B", "GP1BB", "GP9", "PF4", "PPBP", "TUBB1", "TREML1", "GATA1"],
        "description": "megakaryocytes / platelet-producing cells",
    },

    # ── Epithelial ──
    "Epithelial cells": {
        "markers": ["EPCAM", "KRT18", "KRT8", "CDH1", "KRT19", "MUC1", "CLDN4", "TJP1"],
        "description": "epithelial cells",
    },
    "Alveolar type 1": {
        "markers": ["AGER", "PDPN", "HOPX", "CLIC5", "CAV1", "AQP5", "EMP2"],
        "description": "alveolar type 1 pneumocytes",
    },
    "Alveolar type 2": {
        "markers": ["SFTPC", "SFTPA1", "SFTPA2", "SFTPB", "ABCA3", "SLC34A2", "LAMP3"],
        "description": "alveolar type 2 pneumocytes",
    },
    "Basal cells": {
        "markers": ["KRT5", "KRT14", "KRT17", "TP63", "S100A2", "NGFR"],
        "description": "basal epithelial cells",
    },
    "Ciliated cells": {
        "markers": ["FOXJ1", "PIFO", "TPPP3", "SNTN", "RSPH1", "CAPS", "CCDC78", "DNAH5"],
        "description": "multiciliated epithelial cells",
    },
    "Club cells": {
        "markers": ["SCGB1A1", "SCGB3A1", "SCGB3A2", "CYP2F1", "BPIFB1", "MUC5B"],
        "description": "club (Clara) secretory cells",
    },
    "Goblet cells": {
        "markers": ["MUC5AC", "MUC5B", "TFF3", "SPDEF", "AGR2", "FCGBP", "CLCA1"],
        "description": "mucus-secreting goblet cells",
    },
    "Ionocytes": {
        "markers": ["CFTR", "FOXI1", "ATP6V1C2", "ATP6V0D2", "ASCL3"],
        "description": "pulmonary ionocytes",
    },
    "Mesothelial cells": {
        "markers": ["MSLN", "CALB2", "WT1", "UPK3B", "HP", "KRT19", "LRRN4"],
        "description": "mesothelial cells",
    },

    # ── Stromal ──
    "Fibroblasts": {
        "markers": ["COL1A1", "COL1A2", "DCN", "LUM", "FAP", "PDGFRA", "VIM", "FN1", "ACTA2"],
        "description": "fibroblasts / mesenchymal stromal cells",
    },
    "Endothelial cells": {
        "markers": ["PECAM1", "VWF", "CDH5", "FLT1", "KDR", "ENG", "CLDN5", "EMCN"],
        "description": "vascular endothelial cells",
    },
    "Smooth muscle cells": {
        "markers": ["ACTA2", "MYH11", "TAGLN", "CNN1", "DES", "CALD1", "MYL9"],
        "description": "smooth muscle cells",
    },
    "Pericytes": {
        "markers": ["RGS5", "PDGFRB", "NOTCH3", "MCAM", "KCNJ8", "ABCC9"],
        "description": "pericytes / mural cells",
    },
    "Cancer-associated fibroblasts": {
        "markers": ["FAP", "PDPN", "ACTA2", "COL1A1", "POSTN", "MMP11", "CXCL12", "IL6"],
        "description": "cancer-associated fibroblasts (CAFs)",
    },
    "Mesenchymal stem cells": {
        "markers": ["NT5E", "THY1", "ENG", "VCAM1", "LEPR", "CXCL12", "PDGFRA", "PDGFRB"],
        "description": "mesenchymal stem / stromal cells",
    },
    "Lymphatic endothelial": {
        "markers": ["PROX1", "LYVE1", "FLT4", "PDPN", "CCL21", "TFF3"],
        "description": "lymphatic endothelial cells",
    },

    # ── Proliferating ──
    "Proliferating cells": {
        "markers": ["MKI67", "TOP2A", "PCNA", "CDK1", "CCNB1", "TYMS", "MCM2", "BIRC5", "UBE2C"],
        "description": "actively proliferating cells",
    },

    # ── Neural ──
    "Neurons": {
        "markers": ["RBFOX3", "SNAP25", "SYN1", "SYP", "MAP2", "TUBB3", "ENO2", "SLC17A7", "NRGN"],
        "description": "neurons",
    },
    "Excitatory neurons": {
        "markers": ["SLC17A7", "SATB2", "CUX2", "SLC17A6", "CAMK2A", "NRGN", "GRIA1"],
        "description": "excitatory (glutamatergic) neurons",
    },
    "Inhibitory neurons": {
        "markers": ["GAD1", "GAD2", "SLC32A1", "SST", "PVALB", "VIP", "ADARB2"],
        "description": "inhibitory (GABAergic) neurons",
    },
    "Astrocytes": {
        "markers": ["GFAP", "AQP4", "SLC1A3", "S100B", "ALDH1L1", "GJA1", "SOX9"],
        "description": "astrocytes",
    },
    "Oligodendrocytes": {
        "markers": ["MBP", "MOG", "PLP1", "MAG", "OLIG1", "OLIG2", "SOX10"],
        "description": "oligodendrocytes",
    },
    "OPCs": {
        "markers": ["PDGFRA", "CSPG4", "OLIG1", "OLIG2", "SOX10", "GPR17", "NEU4"],
        "description": "oligodendrocyte precursor cells (OPCs)",
    },
    "Microglia": {
        "markers": ["CX3CR1", "P2RY12", "TMEM119", "CSF1R", "AIF1", "HEXB", "TREM2"],
        "description": "microglia",
    },
    "Schwann cells": {
        "markers": ["MPZ", "MBP", "PMP22", "SOX10", "S100B", "EGR2", "PLP1", "CDH19"],
        "description": "Schwann cells (peripheral glia)",
    },
    "Radial glia": {
        "markers": ["PAX6", "SOX2", "NES", "HES1", "HES5", "FABP7", "VIM", "GFAP", "EOMES"],
        "description": "radial glial / neural progenitor cells",
    },

    # ── Stem / Progenitor ──
    "HSCs/Progenitors": {
        "markers": ["CD34", "KIT", "FLT3", "PROM1", "THY1", "CRHBP", "MLLT3", "HLF", "HOPX"],
        "description": "hematopoietic stem and progenitor cells",
    },
    "Erythroid progenitors": {
        "markers": ["GYPA", "GYPB", "HBB", "HBA1", "HBA2", "ALAS2", "KLF1", "TFRC"],
        "description": "erythroid lineage cells",
    },

    # ── Hepatic ──
    "Hepatocytes": {
        "markers": ["ALB", "APOB", "APOA1", "HP", "TF", "TTR", "SERPINA1", "CYP3A4"],
        "description": "hepatocytes",
    },
    "Cholangiocytes": {
        "markers": ["KRT7", "KRT19", "SOX9", "EPCAM", "SPP1", "ANXA4", "FXYD2"],
        "description": "cholangiocytes / bile duct epithelial cells",
    },
    # ── Pancreatic ──
    "Beta cells": {
        "markers": ["INS", "INS1", "INS2", "IAPP", "PCSK1", "PCSK2", "PDX1", "MAFA"],
        "description": "pancreatic beta cells",
    },
    "Alpha cells": {
        "markers": ["GCG", "TTR", "ARX", "MAFB", "IRX2", "LOXL4"],
        "description": "pancreatic alpha cells",
    },
    "Delta cells": {
        "markers": ["SST", "HHEX", "GHSR", "RBP4", "ADCYAP1"],
        "description": "pancreatic delta cells",
    },
    # ── Kidney ──
    "Proximal tubule": {
        "markers": ["SLC34A1", "SLC5A2", "ALDOB", "AQP1", "LRP2", "CUBN"],
        "description": "kidney proximal tubule cells",
    },
    "Collecting duct": {
        "markers": ["AQP2", "AQP3", "SLC12A1", "SCNN1A", "KRT8", "KRT18"],
        "description": "kidney collecting duct cells",
    },
    "Podocytes": {
        "markers": ["NPHS1", "NPHS2", "PODXL", "WT1", "SYNPO"],
        "description": "glomerular podocytes",
    },
    # ── Muscle ──
    "Skeletal muscle": {
        "markers": ["MYH1", "MYH2", "MYH7", "ACTA1", "TTN", "MYLK2"],
        "description": "skeletal muscle cells",
    },
    "Cardiomyocytes": {
        "markers": ["TNNT2", "MYH6", "MYH7", "ACTC1", "RYR2", "MYL2"],
        "description": "cardiomyocytes",
    },
    # ── Endocrine / Other ──
    "Adipocytes": {
        "markers": ["ADIPOQ", "LEP", "FABP4", "PLIN1", "LPL"],
        "description": "adipocytes",
    },
    "Melanocytes": {
        "markers": ["MITF", "MLANA", "TYR", "DCT", "PMEL"],
        "description": "melanocytes",
    },
    # ── Gastrointestinal ──
    "Enterocytes": {
        "markers": ["FABP2", "ALPI", "SLC5A1", "SI", "VIL1", "APOA1", "APOA4"],
        "description": "intestinal enterocytes",
    },
    "Paneth cells": {
        "markers": ["DEFA5", "DEFA6", "LYZ", "REG3A", "MMP7", "ITLN2"],
        "description": "Paneth cells (intestinal innate defense)",
    },
    "Tuft cells": {
        "markers": ["POU2F3", "TRPM5", "DCLK1", "GFI1B", "AVIL", "SH2D6"],
        "description": "tuft (chemosensory) cells",
    },
    # ── Thymic ──
    "Thymic epithelial": {
        "markers": ["KRT5", "KRT14", "AIRE", "FOXN1", "PSMB11", "CCL25", "DLL4"],
        "description": "thymic epithelial cells",
    },
    # ── Red blood cells ──
    "Erythrocytes": {
        "markers": ["HBB", "HBA1", "HBA2", "HBD", "SLC4A1", "ANK1", "GYPA", "SPTA1"],
        "description": "mature red blood cells / erythrocytes",
    },

    # ── Neuroendocrine / Merkel ──
    "Neuroendocrine cells": {
        "markers": ["CHGA", "CHGB", "SYP", "ENO2", "NCAM1", "INSM1", "ASCL1", "NHLH1",
                    "MDK", "CCK", "NPY", "VIP", "GRP", "SCG2", "DDC", "BEX1"],
        "description": "neuroendocrine cells",
    },
    "Merkel cells": {
        "markers": ["ATOH1", "ISL1", "SOX2", "KRT20", "KRT8", "KRT18", "NHLH1", "CHGA",
                    "MDK", "CCK", "NPY", "VIP", "ENO1", "S100A1"],
        "description": "Merkel cells (neuroendocrine touch receptors)",
    },

    # ── Spinal cord / motor neurons ──
    "Motor neurons": {
        "markers": ["CHAT", "ISL1", "ISL2", "MNX1", "SLC18A3", "SLC5A7", "PRPH", "LHX3"],
        "description": "motor neurons",
    },
    "Spinal interneurons": {
        "markers": ["PAX2", "EN1", "EVX1", "LBX1", "LHX1", "LHX5", "GRIA1", "GAD1",
                    "GAD2", "SLC32A1", "GBX1", "ELMO1"],
        "description": "spinal cord interneurons",
    },
    "Dorsal horn neurons": {
        "markers": ["LBX1", "PAX2", "TLX3", "DRG11", "SST", "TAC1", "PVALB", "CALB1",
                    "CALB2", "NPY", "NOS1AP"],
        "description": "dorsal horn sensory neurons",
    },

    # ── Hippocampal / DG neurons ──
    "Granule cells": {
        "markers": ["PROX1", "CALB2", "NEUROD6", "NEUROD2", "BHLHE22", "BCL11B",
                    "SEMA5A", "NRGN", "CNTNAP2", "CNTNAP5A"],
        "description": "hippocampal dentate gyrus granule cells",
    },
    "Immature neurons": {
        "markers": ["DCX", "NEUROD1", "NEUROD2", "NEUROD6", "TUBB3", "STMN2",
                    "CD24", "SNAP25", "NREP", "SOX11", "NFIB", "RTN1"],
        "description": "immature / migrating neurons",
    },
    "Intermediate progenitors": {
        "markers": ["EOMES", "TBR2", "NEUROG2", "NEUROG1", "HES6", "NHLH1",
                    "GADD45G", "DLL1", "ASCL1", "INSM1", "MKI67"],
        "description": "intermediate neural progenitors",
    },

    # ── Retinal ──
    "Retinal ganglion cells": {
        "markers": ["POU4F1", "POU4F2", "ISL1", "RBPMS", "SNCG", "THY1", "GAP43",
                    "NEFL", "NEFM"],
        "description": "retinal ganglion cells",
    },
    "Photoreceptors": {
        "markers": ["RHO", "OPN1SW", "OPN1MW", "NRL", "CRX", "NR2E3", "RCVRN",
                    "PDE6A", "GNAT1", "ARR3"],
        "description": "photoreceptor cells (rods and cones)",
    },
    "Bipolar cells": {
        "markers": ["VSX2", "OTX2", "VSX1", "TRPM1", "GRM6", "PRKCA", "CABP5"],
        "description": "retinal bipolar cells",
    },
    "Müller glia": {
        "markers": ["RLBP1", "GLUL", "SLC1A3", "VIM", "SOX9", "CLU", "DKK3",
                    "AQP4", "APOE"],
        "description": "Müller glial cells",
    },
    "Amacrine cells": {
        "markers": ["GAD1", "GAD2", "SLC32A1", "SLC6A9", "TFAP2A", "TFAP2B",
                    "PAX6", "CHAT", "CALB1"],
        "description": "retinal amacrine cells",
    },

    # ── Pituitary ──
    "Somatotrophs": {
        "markers": ["GH1", "GH2", "GHRHR", "DLK1", "RBP4", "OTOS", "IGSF1"],
        "description": "pituitary somatotroph cells (growth hormone-producing)",
    },
    "Lactotrophs": {
        "markers": ["PRL", "DRD2", "GATA2", "ESR1", "TBX19"],
        "description": "pituitary lactotroph cells (prolactin-producing)",
    },
    "Corticotrophs": {
        "markers": ["POMC", "TBX19", "TPIT", "PCSK1", "CRH", "CRHR1"],
        "description": "pituitary corticotroph cells (ACTH-producing)",
    },
    "Gonadotrophs": {
        "markers": ["FSHB", "LHB", "CGA", "NR5A1", "GNRHR", "FOXL2"],
        "description": "pituitary gonadotroph cells (FSH/LH-producing)",
    },
    "Thyrotrophs": {
        "markers": ["TSHB", "CGA", "POU1F1", "GATA2", "RXRG"],
        "description": "pituitary thyrotroph cells (TSH-producing)",
    },

    # ── Urothelial / prostate / urethral ──
    "Urothelial cells": {
        "markers": ["UPK1A", "UPK1B", "UPK2", "UPK3A", "UPK3B", "KRT20", "PSCA",
                    "KRT8", "KRT18", "SLC12A2"],
        "description": "urothelial / transitional epithelial cells",
    },
    "Luminal epithelial": {
        "markers": ["KRT8", "KRT18", "KRT19", "MUC1", "CDH1", "EPCAM", "GATA3",
                    "FOXA1", "ESR1", "PGR", "AR"],
        "description": "luminal epithelial cells",
    },
    "Myoepithelial cells": {
        "markers": ["KRT5", "KRT14", "ACTA2", "TAGLN", "MYH11", "CNN1", "TP63",
                    "SERPINB5", "OXTR"],
        "description": "myoepithelial / basal cells",
    },
    "Prostate epithelial": {
        "markers": ["MSMB", "PBSN", "KLK3", "KLK2", "NKX3-1", "AR", "ACPP",
                    "SLC45A3", "TGM4", "DEFB50"],
        "description": "prostate epithelial cells",
    },

    # ── Mammary / breast-specific ──
    "Mammary secretory": {
        "markers": ["CSN2", "CSN3", "LALBA", "LTF", "BTN1A1", "XDH", "ELF5"],
        "description": "mammary secretory epithelial cells",
    },

    # ── Reproductive ──
    "Leydig cells": {
        "markers": ["CYP17A1", "CYP11A1", "HSD3B1", "STAR", "INSL3", "CYP19A1"],
        "description": "Leydig cells (testosterone-producing)",
    },
    "Sertoli cells": {
        "markers": ["SOX9", "WT1", "AMH", "CLDN11", "GATA4", "CLU", "INHA"],
        "description": "Sertoli cells (testicular supporting cells)",
    },

    # ── Dental / odontogenic ──
    "Odontoblasts": {
        "markers": ["DSPP", "DMP1", "COL1A1", "ALPL", "PHEX", "RUNX2", "IBSP",
                    "SPARC", "BGLAP"],
        "description": "odontoblasts / dentin-secreting cells",
    },
    "Ameloblasts": {
        "markers": ["AMELX", "ENAM", "AMBN", "KLK4", "MMP20", "SLC24A4"],
        "description": "ameloblasts (enamel-producing cells)",
    },
    "Dental epithelial": {
        "markers": ["SHH", "PITX2", "PAX9", "MSX1", "DLX2", "BMP4", "FGF4"],
        "description": "dental epithelial progenitor cells",
    },

    # ── Osteoblast / chondrocyte ──
    "Osteoblasts": {
        "markers": ["BGLAP", "RUNX2", "SP7", "ALPL", "COL1A1", "IBSP", "SPP1",
                    "DMP1", "SOST"],
        "description": "osteoblasts / bone-forming cells",
    },
    "Chondrocytes": {
        "markers": ["SOX9", "COL2A1", "ACAN", "COL9A1", "COL11A1", "COMP",
                    "MATN3", "FMOD"],
        "description": "chondrocytes / cartilage cells",
    },

    # ── Functional / state-based signatures (catch stress, IFN, etc.) ──
    "Stress-response cells": {
        "markers": ["HSPA1A", "HSPA1B", "HSP90AA1", "HSP90AB1", "HSPH1", "HSPB1",
                    "HSPE1", "DNAJB1", "DNAJB6", "ATF3", "FOS", "JUN", "JUNB"],
        "description": "cells in heat-shock / stress response state",
    },
    "IFN-stimulated cells": {
        "markers": ["ISG15", "IFIT1", "IFIT3", "IFITM1", "IFITM3", "MX1", "MX2",
                    "OAS1", "STAT1", "IRF7", "BST2", "IFI6", "IFI27", "IFI44L"],
        "description": "interferon-stimulated / antiviral response cells",
    },
    "MHC-II high APCs": {
        "markers": ["HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1", "HLA-DQA1",
                    "HLA-DQB1", "CD74", "H2-AA", "H2-AB1", "H2-EB1", "CIITA"],
        "description": "MHC-II-high antigen-presenting cells",
    },
    "Cycling cells": {
        "markers": ["MKI67", "TOP2A", "PCNA", "CDK1", "CCNB1", "CCNB2", "TYMS",
                    "MCM2", "MCM5", "BIRC5", "UBE2C", "STMN1", "TUBA1B", "TUBB"],
        "description": "actively cycling cells (S/G2/M phase)",
    },

    # ── Endoderm / stem-cell derived ──
    "Definitive endoderm": {
        "markers": ["SOX17", "FOXA2", "CXCR4", "GATA4", "GATA6", "GSC", "CER1",
                    "HHEX", "MIXL1"],
        "description": "definitive endoderm cells",
    },
    "Pluripotent stem cells": {
        "markers": ["POU5F1", "SOX2", "NANOG", "ZFP42", "DPPA4", "LIN28A",
                    "DNMT3B", "L1TD1", "PODXL", "SALL4"],
        "description": "pluripotent stem cells (ESC/iPSC)",
    },
    "Mesoderm progenitors": {
        "markers": ["MESP1", "MESP2", "MIXL1", "TBX6", "MSGN1", "WNT3A", "CDX2",
                    "HAND1", "BMP4"],
        "description": "mesoderm progenitor cells",
    },

    # ── Capillary / vascular subtypes ──
    "Capillary endothelial": {
        "markers": ["CA4", "CD36", "FABP4", "FABP5", "GPIHBP1", "RBP7", "BTNL9",
                    "CD300LG", "CLDN5"],
        "description": "capillary endothelial cells",
    },
    "Venous endothelial": {
        "markers": ["ACKR1", "CLU", "SELE", "SELP", "VWF", "CCL14", "PLVAP",
                    "CD74", "HLA-DRA"],
        "description": "venous endothelial cells",
    },
    "Tip cells": {
        "markers": ["ESM1", "APLN", "ANGPT2", "DLL4", "KDR", "CXCR4", "SPARC",
                    "COL4A1", "COL4A2"],
        "description": "tip / angiogenic endothelial cells",
    },
}

# Context-aware priors: if dataset text mentions these keywords, boost relevant cell types.
CONTEXT_PRIORS = {
    "lung": ["Alveolar type 1", "Alveolar type 2", "Basal cells", "Ciliated cells", "Club cells",
             "Goblet cells", "Ionocytes", "Epithelial cells", "Alveolar macrophages",
             "Endothelial cells", "Fibroblasts", "Macrophages", "NK cells", "CD8+ T cells"],
    "airway": ["Ciliated cells", "Club cells", "Goblet cells", "Basal cells", "Ionocytes",
               "Epithelial cells", "Macrophages"],
    "brain": ["Neurons", "Excitatory neurons", "Inhibitory neurons", "Astrocytes",
              "Oligodendrocytes", "OPCs", "Microglia", "Radial glia"],
    "cortex": ["Excitatory neurons", "Inhibitory neurons", "Astrocytes",
               "Oligodendrocytes", "OPCs", "Microglia"],
    "kidney": ["Proximal tubule", "Collecting duct", "Podocytes", "Endothelial cells",
               "Fibroblasts", "Mesothelial cells"],
    "liver": ["Hepatocytes", "Cholangiocytes", "Endothelial cells", "Kupffer cells",
              "Macrophages", "NK cells", "Lymphatic endothelial"],
    "pancreas": ["Beta cells", "Alpha cells", "Delta cells", "Endothelial cells",
                 "Fibroblasts", "Macrophages"],
    "bone marrow": ["HSCs/Progenitors", "Erythroid progenitors", "Monocytes", "Neutrophils",
                    "B cells", "Megakaryocytes", "Erythrocytes", "Plasma cells"],
    "hematopoiesis": ["HSCs/Progenitors", "Erythroid progenitors", "Monocytes", "Neutrophils",
                      "B cells", "Megakaryocytes", "Erythrocytes"],
    "hemato": ["HSCs/Progenitors", "Erythroid progenitors", "Monocytes", "Neutrophils",
               "B cells", "Megakaryocytes"],
    "pbmc": ["CD8+ T cells", "CD4+ T cells", "NK cells", "B cells", "Monocytes",
             "Dendritic cells", "Plasma cells", "Regulatory T cells", "Gamma-delta T cells"],
    "blood": ["CD8+ T cells", "CD4+ T cells", "NK cells", "B cells", "Monocytes",
              "Dendritic cells", "Neutrophils", "Erythrocytes", "Megakaryocytes"],
    "tumor": ["Epithelial cells", "Cancer-associated fibroblasts", "Fibroblasts",
              "Endothelial cells", "Macrophages", "CD8+ T cells", "NK cells",
              "Exhausted T cells", "Regulatory T cells"],
    "cancer": ["Epithelial cells", "Cancer-associated fibroblasts", "Fibroblasts",
               "Endothelial cells", "Macrophages", "CD8+ T cells", "NK cells",
               "Exhausted T cells", "Regulatory T cells"],
    "carcinoma": ["Epithelial cells", "Cancer-associated fibroblasts", "Macrophages",
                  "CD8+ T cells", "Exhausted T cells", "Regulatory T cells"],
    "melanoma": ["Melanocytes", "Macrophages", "CD8+ T cells", "NK cells",
                 "Exhausted T cells", "Fibroblasts"],
    "leukemia": ["HSCs/Progenitors", "B cells", "Monocytes", "CD8+ T cells",
                 "Erythroid progenitors", "Megakaryocytes"],
    "lymphoma": ["B cells", "CD8+ T cells", "CD4+ T cells", "NK cells",
                 "Macrophages", "Fibroblasts"],
    "skin": ["Basal cells", "Melanocytes", "Langerhans cells", "Fibroblasts",
             "Endothelial cells", "Macrophages", "CD8+ T cells"],
    "intestin": ["Enterocytes", "Goblet cells", "Paneth cells", "Tuft cells",
                 "Epithelial cells", "Macrophages", "CD8+ T cells"],
    "colon": ["Enterocytes", "Goblet cells", "Epithelial cells", "Macrophages",
              "Fibroblasts", "CD8+ T cells"],
    "thymus": ["Thymic epithelial", "CD4+ T cells", "CD8+ T cells", "Dendritic cells",
               "Macrophages"],
    "muscle": ["Skeletal muscle", "Smooth muscle cells", "Fibroblasts", "Endothelial cells",
               "Macrophages", "Pericytes"],
    "heart": ["Cardiomyocytes", "Fibroblasts", "Endothelial cells", "Pericytes",
              "Smooth muscle cells", "Macrophages"],
    "spleen": ["B cells", "CD8+ T cells", "CD4+ T cells", "Macrophages", "NK cells",
               "Dendritic cells", "Erythrocytes"],
    "eye": ["Neurons", "Astrocytes", "Pericytes", "Endothelial cells", "Macrophages"],
    "peripheral nerve": ["Schwann cells", "Neurons", "Fibroblasts", "Endothelial cells"],
    "embryo": ["Radial glia", "HSCs/Progenitors", "Mesenchymal stem cells",
               "Proliferating cells", "Epithelial cells"],
    "development": ["Radial glia", "HSCs/Progenitors", "Mesenchymal stem cells",
                    "Proliferating cells"],
    "fetal": ["Radial glia", "HSCs/Progenitors", "Erythroid progenitors",
              "Mesenchymal stem cells", "Proliferating cells"],
    "immune": ["CD8+ T cells", "CD4+ T cells", "NK cells", "B cells", "Monocytes",
               "Macrophages", "Dendritic cells", "Neutrophils", "Mast cells"],
    "lps": ["Macrophages", "Monocytes", "Microglia", "Dendritic cells", "Neutrophils"],
    "aging": ["CD8+ T cells", "CD4+ T cells", "NK cells", "B cells", "Monocytes",
              "Macrophages", "HSCs/Progenitors"],
    "pituitary": ["Somatotrophs", "Lactotrophs", "Corticotrophs", "Gonadotrophs",
                  "Thyrotrophs", "Endothelial cells", "Fibroblasts"],
    "retina": ["Retinal ganglion cells", "Photoreceptors", "Bipolar cells",
               "Müller glia", "Amacrine cells", "Astrocytes", "Microglia"],
    "hippocampus": ["Granule cells", "Immature neurons", "Intermediate progenitors",
                    "Radial glia", "Astrocytes", "Microglia", "Oligodendrocytes",
                    "Inhibitory neurons", "Excitatory neurons"],
    "dentate": ["Granule cells", "Immature neurons", "Intermediate progenitors",
                "Radial glia", "Astrocytes", "Oligodendrocytes"],
    "spinal": ["Motor neurons", "Spinal interneurons", "Dorsal horn neurons",
               "Astrocytes", "Oligodendrocytes", "OPCs", "Microglia",
               "Excitatory neurons", "Inhibitory neurons"],
    "spine": ["Motor neurons", "Spinal interneurons", "Dorsal horn neurons",
              "Astrocytes", "Oligodendrocytes", "OPCs", "Microglia"],
    "prostate": ["Prostate epithelial", "Luminal epithelial", "Basal cells",
                 "Smooth muscle cells", "Fibroblasts", "Endothelial cells"],
    "urethr": ["Urothelial cells", "Luminal epithelial", "Basal cells",
               "Myoepithelial cells", "Smooth muscle cells", "Fibroblasts"],
    "bladder": ["Urothelial cells", "Luminal epithelial", "Smooth muscle cells",
                "Fibroblasts", "Endothelial cells"],
    "breast": ["Luminal epithelial", "Myoepithelial cells", "Mammary secretory",
               "Cancer-associated fibroblasts", "Endothelial cells", "Macrophages",
               "CD8+ T cells", "Adipocytes"],
    "mammary": ["Luminal epithelial", "Myoepithelial cells", "Mammary secretory",
                "Fibroblasts", "Adipocytes", "Endothelial cells"],
    "organoid": ["Radial glia", "Immature neurons", "Intermediate progenitors",
                 "Excitatory neurons", "Inhibitory neurons", "Astrocytes",
                 "Pluripotent stem cells", "Proliferating cells"],
    "forebrain": ["Radial glia", "Immature neurons", "Intermediate progenitors",
                  "Excitatory neurons", "Inhibitory neurons", "OPCs", "Astrocytes"],
    "thalamus": ["Excitatory neurons", "Inhibitory neurons", "Astrocytes",
                  "Oligodendrocytes", "OPCs", "Microglia"],
    "esc": ["Pluripotent stem cells", "Definitive endoderm", "Mesoderm progenitors",
            "Proliferating cells"],
    "hesc": ["Pluripotent stem cells", "Definitive endoderm", "Mesoderm progenitors",
             "Proliferating cells"],
    "ipsc": ["Pluripotent stem cells", "Definitive endoderm", "Mesoderm progenitors",
             "Proliferating cells"],
    "stem cell": ["Pluripotent stem cells", "HSCs/Progenitors", "Mesenchymal stem cells",
                  "Radial glia", "Proliferating cells"],
    "differentiation": ["Pluripotent stem cells", "Definitive endoderm",
                        "Mesoderm progenitors", "Intermediate progenitors",
                        "Immature neurons", "Radial glia"],
    "endoderm": ["Definitive endoderm", "Hepatocytes", "Beta cells", "Alpha cells",
                 "Enterocytes", "Goblet cells", "Cholangiocytes"],
    "als": ["Motor neurons", "Spinal interneurons", "Astrocytes", "Microglia",
            "Oligodendrocytes", "OPCs"],
    "aorta": ["Smooth muscle cells", "Endothelial cells", "Fibroblasts",
              "Macrophages", "CD8+ T cells", "Pericytes"],
    "vessel": ["Endothelial cells", "Smooth muscle cells", "Pericytes",
               "Fibroblasts", "Capillary endothelial", "Venous endothelial"],
    "lymph node": ["B cells", "CD8+ T cells", "CD4+ T cells", "Dendritic cells",
                   "Macrophages", "NK cells", "Plasma cells"],
    "mesenteri": ["B cells", "CD4+ T cells", "CD8+ T cells", "Dendritic cells",
                  "Macrophages", "NK cells", "ILC cells"],
    "tooth": ["Odontoblasts", "Ameloblasts", "Dental epithelial", "Fibroblasts",
              "Mesenchymal stem cells", "Endothelial cells"],
    "dent": ["Odontoblasts", "Ameloblasts", "Dental epithelial", "Fibroblasts",
             "Mesenchymal stem cells"],
    "alzheim": ["Microglia", "Astrocytes", "Neurons", "Oligodendrocytes",
                "OPCs", "Endothelial cells"],
    "autism": ["Microglia", "Astrocytes", "Neurons", "Excitatory neurons",
               "Inhibitory neurons", "Oligodendrocytes"],
    "hepatoblastoma": ["Hepatocytes", "Cholangiocytes", "Endothelial cells",
                       "Macrophages", "Fibroblasts", "Mesenchymal stem cells"],
    "stomach": ["Epithelial cells", "Enterocytes", "Goblet cells",
                "Macrophages", "Fibroblasts", "CD8+ T cells"],
    "gastric": ["Epithelial cells", "Enterocytes", "Goblet cells",
                "Macrophages", "Fibroblasts", "CD8+ T cells"],
    "myeloma": ["Plasma cells", "B cells", "CD8+ T cells", "NK cells",
                "Monocytes", "Macrophages", "Erythroid progenitors"],
    "myeloid": ["Monocytes", "Macrophages", "Dendritic cells", "Neutrophils",
                "Mast cells", "Plasmacytoid DCs"],
    "lsk": ["HSCs/Progenitors", "Erythroid progenitors", "Megakaryocytes",
            "Monocytes", "Proliferating cells"],
    "neuroendocrin": ["Neuroendocrine cells", "Merkel cells"],
    "merkel": ["Merkel cells", "Neuroendocrine cells", "CD8+ T cells"],
    "interferon": ["IFN-stimulated cells", "Monocytes", "Macrophages",
                   "HSCs/Progenitors"],
    "cd45": ["CD8+ T cells", "CD4+ T cells", "NK cells", "Macrophages",
             "Monocytes", "Dendritic cells", "B cells", "Neutrophils"],
    "leukocyte": ["CD8+ T cells", "CD4+ T cells", "NK cells", "Macrophages",
                  "Monocytes", "Dendritic cells", "B cells", "Neutrophils"],
    "endothelial": ["Endothelial cells", "Capillary endothelial", "Venous endothelial",
                    "Lymphatic endothelial", "Tip cells", "Pericytes"],
    "strom": ["Fibroblasts", "Smooth muscle cells", "Pericytes",
              "Endothelial cells", "Mesenchymal stem cells", "Adipocytes"],
    "islet": ["Beta cells", "Alpha cells", "Delta cells", "Endothelial cells",
              "Macrophages"],
    "tumor infiltrat": ["CD8+ T cells", "CD4+ T cells", "NK cells",
                        "Macrophages", "Dendritic cells", "Regulatory T cells",
                        "Exhausted T cells", "Monocytes"],
}


def _load_signature_db(path: str) -> Dict[str, Dict[str, List[str]]]:
    """Load additional marker signatures from a JSON file."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        logger.warning(f"Signature DB not found: {path}")
        return {}
    try:
        with open(p) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.warning(f"Invalid signature DB format: {path}")
            return {}
        return data
    except Exception as e:
        logger.warning(f"Failed to load signature DB: {e}")
        return {}


def annotate_cluster(
    top_markers: List[str],
    dataset_text: str,
    cluster_id: int,
    n_cells: int,
    resolution: float,
    signatures: Dict[str, Dict[str, List[str]]],
    dataset_id: str = "",
) -> Tuple[str, str, float]:
    """
    Annotate a cluster based on its top marker genes.
    
    Returns: (cell_type_name, description_text, confidence_score)
    """
    best_type = None
    best_score = 0.0
    best_matched = []
    second_best_type = None
    second_best_score = 0.0

    top_set = set([m.upper() for m in top_markers[:30]])  # use top 30 markers
    # Also check top 15 (higher weight — more specific markers)
    top15_set = set([m.upper() for m in top_markers[:15]])

    context = dataset_text.lower()
    context_hits = set()
    for kw, types in CONTEXT_PRIORS.items():
        if kw in context:
            context_hits.update(types)

    scores = []
    for ct_name, ct_info in signatures.items():
        sig_markers = set([m.upper() for m in ct_info["markers"]])
        overlap = top_set & sig_markers
        overlap_top15 = top15_set & sig_markers
        # Weighted scoring: top-15 markers count more (they're more specific)
        if len(sig_markers) > 0:
            denom = min(len(sig_markers), 8)
            base = len(overlap) / denom
            # Extra weight for markers in top 15 (higher rank = more discriminative)
            rank_bonus = 0.05 * len(overlap_top15)
        else:
            base = 0
            rank_bonus = 0
        # Context prior bonus
        bonus = 0.1 if ct_name in context_hits else 0.0
        score = min(base + bonus + rank_bonus, 1.0)
        scores.append((score, ct_name, sorted(overlap)))

    scores.sort(reverse=True)
    if scores:
        best_score, best_type, best_matched = scores[0]
    if len(scores) > 1:
        second_best_score, second_best_type, _ = scores[1]

    # Lower threshold from 0.25 → 0.15 to catch more valid annotations
    # Use margin check: if top two scores are very close, flag lower confidence
    if best_score >= 0.15 and best_type is not None:
        ct_desc = signatures[best_type]["description"]
        matched_str = ", ".join(best_matched[:5])
        
        # Adjust confidence if margin is small (ambiguous assignment)
        margin = best_score - second_best_score
        effective_conf = best_score if margin > 0.1 else best_score * 0.85
        
        # Build cluster-specific description combining dataset context + cell type
        # Two-layer: dataset-level metadata (organism/tissue/disease) + cluster-level annotation
        tissue_context = _extract_context(dataset_text, dataset_id)
        ctx = _parse_structured_context(dataset_text, dataset_id)
        if best_score >= 0.50:
            prefix = "Sub-population of"
        elif best_score >= 0.30:
            prefix = "Putative"
        else:
            prefix = "Candidate"
        description = (
            f"In {ctx['organism']} {ctx['tissue']}"
            + (f" ({ctx['disease']})" if ctx['disease'] != 'healthy' else "")
            + f": {prefix} {ct_desc} "
            f"(Leiden cluster {cluster_id}, n={n_cells} cells, "
            f"res={resolution:.2f}) identified by expression of {matched_str}. "
            f"Source: {ctx['context_summary']}"
        )
        return best_type, description, effective_conf
    else:
        # Unknown cluster — use top markers for description
        top_5 = ", ".join(top_markers[:5])
        tissue_context = _extract_context(dataset_text, dataset_id)
        ctx = _parse_structured_context(dataset_text, dataset_id)
        description = (
            f"In {ctx['organism']} {ctx['tissue']}"
            + (f" ({ctx['disease']})" if ctx['disease'] != 'healthy' else "")
            + f": Uncharacterized cell population "
            f"(Leiden cluster {cluster_id}, n={n_cells} cells, "
            f"res={resolution:.2f}) with high expression of {top_5}. "
            f"Source: {ctx['context_summary']}"
        )
        return "Unknown", description, best_score


def _parse_structured_context(dataset_text: str, dataset_id: str = "") -> dict:
    """Parse organism, tissue, disease from dataset text and ID.
    
    Returns dict with keys: organism, tissue, disease, context_summary
    """
    text_lower = dataset_text.lower()
    id_lower = dataset_id.lower()
    
    # --- Organism ---
    organism = "unknown organism"
    if any(x in text_lower for x in ["human", "hm", "hesc", "pbmc"]):
        organism = "human"
    elif "hm" in id_lower and "mm" not in id_lower:
        organism = "human"
    elif any(x in text_lower for x in ["mouse", "murine", "mus musculus"]):
        organism = "mouse"
    elif "mm" in id_lower:
        organism = "mouse"
    # Fallback: check for human/mouse gene naming convention in text
    if organism == "unknown organism":
        if any(x in text_lower for x in ["patients", "donors", "clinical"]):
            organism = "human"
        elif any(x in text_lower for x in ["c57bl", "balb", "nod", "transgenic mice"]):
            organism = "mouse"
    
    # --- Tissue ---
    tissue_map = [
        (["lung", "pulmonary", "alveolar", "airway", "bronch"], "lung"),
        (["brain", "cortex", "cortical", "cerebr", "hippocampus", "dentate", "forebrain",
          "thalamus"], "brain"),
        (["spinal cord", "spine", "spinal"], "spinal cord"),
        (["liver", "hepat"], "liver"),
        (["kidney", "renal", "nephro"], "kidney"),
        (["pancrea", "islet"], "pancreas"),
        (["bone marrow", "bm ", "marrow"], "bone marrow"),
        (["blood", "pbmc", "peripheral"], "peripheral blood"),
        (["skin", "cutaneous", "derma"], "skin"),
        (["intestin", "colon", "gut", "ileum", "jejunum", "duodenum"], "intestine"),
        (["breast", "mammary"], "breast"),
        (["heart", "cardiac", "myocard"], "heart"),
        (["retina", "retinal", "eye", "ocular"], "retina"),
        (["pituitary"], "pituitary gland"),
        (["thymus", "thymic"], "thymus"),
        (["spleen", "splenic"], "spleen"),
        (["aort", "vessel", "vascul"], "vasculature"),
        (["prostate", "prostatic"], "prostate"),
        (["urethr", "bladder", "urothel"], "urethra/bladder"),
        (["stomach", "gastric"], "stomach"),
        (["lymph node", "mesenteric"], "lymph node"),
        (["tooth", "dental", "teeth"], "dental tissue"),
        (["muscle", "skeletal"], "muscle"),
        (["organoid"], "organoid"),
    ]
    tissue = "tissue"
    for keywords, t_name in tissue_map:
        if any(kw in text_lower or kw in id_lower for kw in keywords):
            tissue = t_name
            break
    
    # --- Disease ---
    disease_map = [
        (["cancer", "tumor", "tumour", "carcinoma", "sarcoma", "malignant",
          "neoplasm", "oncol"], "cancer"),
        (["melanoma"], "melanoma"),
        (["leukemia", "leukaemia", "all ", "aml", "cml"], "leukemia"),
        (["lymphoma"], "lymphoma"),
        (["myeloma", " mm "], "multiple myeloma"),
        (["glioblastoma", "glioma", "gbm"], "glioma"),
        (["metastas", "metastic"], "metastatic cancer"),
        (["hepatoblastoma"], "hepatoblastoma"),
        (["alzheim"], "Alzheimer's disease"),
        (["parkinsons", "parkinson"], "Parkinson's disease"),
        (["als", "amyotrophic"], "ALS"),
        (["autism", "asd"], "autism spectrum disorder"),
        (["aging", "aged"], "aging"),
        (["injur", "sci "], "injury"),
        (["fibrosis", "fibrotic"], "fibrosis"),
        (["cystic fibrosis", "cftr"], "cystic fibrosis"),
        (["inflamm", "lps"], "inflammation"),
    ]
    disease = "healthy"
    for keywords, d_name in disease_map:
        if any(kw in text_lower or kw in id_lower for kw in keywords):
            disease = d_name
            break
    
    # --- Context summary ---
    sentences = [s.strip() for s in dataset_text.split(". ") if s.strip()]
    context_summary = ". ".join(sentences[:2]).strip().rstrip(".")
    context_summary = context_summary.replace("Single-cell RNA sequencing of ", "")
    if len(context_summary) > 120:
        context_summary = context_summary[:117].rsplit(" ", 1)[0] + "..."
    
    return {
        "organism": organism,
        "tissue": tissue,
        "disease": disease,
        "context_summary": context_summary,
    }


def _extract_context(dataset_text: str, dataset_id: str = "") -> str:
    """Build two-layer context string: organism + tissue + disease + summary."""
    ctx = _parse_structured_context(dataset_text, dataset_id)
    parts = []
    parts.append(f"{ctx['organism']} {ctx['tissue']}")
    if ctx['disease'] != "healthy":
        parts.append(f"({ctx['disease']})")
    parts.append(f"— {ctx['context_summary']}")
    return " ".join(parts)


def choose_resolution(n_cells: int, base_resolution: float, mode: str) -> float:
    """Choose Leiden clustering resolution based on dataset size.

    v6.1: Increased resolutions ~2× to produce finer clusters, addressing the
    text-cell granularity gap (67% within-group variance). More clusters → more
    unique text descriptions → between-group variance increases.

    Previous: 0.6/0.8/1.0/1.2 → New: 1.2/1.5/2.0/2.5
    """
    if mode == "fixed":
        return base_resolution
    if n_cells < 1000:
        return 1.2
    if n_cells < 5000:
        return 1.5
    if n_cells < 15000:
        return 2.0
    return 2.5


def cluster_and_annotate(
    adata: sc.AnnData,
    dataset_id: str,
    dataset_text: str,
    resolution: float = 0.8,
    min_cluster_size: int = 20,
    signatures: Dict[str, Dict[str, List[str]]] = None,
) -> Dict:
    """
    Cluster an AnnData and annotate each cluster.
    
    Returns dict mapping cluster_id → {cell_type, text, confidence, n_cells, markers}
    """
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    adata_work = adata.copy()
    
    # PCA → neighbors → Leiden
    try:
        if "X_pca" not in adata_work.obsm:
            n_comps = min(30, adata_work.shape[0] - 1, adata_work.shape[1] - 1)
            if n_comps < 5:
                logger.warning(f"[{dataset_id}] Too small for clustering ({adata_work.shape})")
                return {}
            sc.tl.pca(adata_work, n_comps=n_comps)
        
        sc.pp.neighbors(adata_work, n_neighbors=15, n_pcs=min(30, adata_work.obsm["X_pca"].shape[1]))
        sc.tl.leiden(adata_work, resolution=resolution, key_added="leiden")
    except Exception as e:
        logger.warning(f"[{dataset_id}] Clustering failed: {e}")
        return {}
    
    clusters = adata_work.obs["leiden"].unique()
    logger.info(f"[{dataset_id}] Found {len(clusters)} clusters (resolution={resolution})")
    
    # Rank genes per cluster
    try:
        sc.tl.rank_genes_groups(adata_work, "leiden", method="wilcoxon", n_genes=50)
    except Exception as e:
        logger.warning(f"[{dataset_id}] Marker gene ranking failed: {e}")
        return {}
    
    cluster_annotations = {}
    signatures = signatures or CELL_TYPE_SIGNATURES
    
    for cluster_id in sorted(clusters, key=int):
        mask = adata_work.obs["leiden"] == cluster_id
        n_cells = int(mask.sum())
        
        if n_cells < min_cluster_size:
            continue
        
        # Get top marker genes for this cluster
        try:
            top_markers = list(adata_work.uns["rank_genes_groups"]["names"][cluster_id][:30])
        except (KeyError, IndexError):
            try:
                # Different scanpy versions store results differently
                result = adata_work.uns["rank_genes_groups"]
                gene_names = [result["names"][i][int(cluster_id)] for i in range(min(30, len(result["names"])))]
                top_markers = gene_names
            except Exception:
                continue
        
        cell_type, description, confidence = annotate_cluster(
            top_markers, dataset_text, int(cluster_id), n_cells, resolution, signatures,
            dataset_id=dataset_id,
        )
        
        cluster_annotations[str(cluster_id)] = {
            "cell_type": cell_type,
            "text": description,
            "confidence": round(float(confidence), 3),
            "n_cells": int(n_cells),
            "top_markers": top_markers[:10],
            "cell_indices": [int(x) for x in np.where(mask)[0]],
            "label_source": "signature" if cell_type != "Unknown" else "marker_only",
        }
        
        logger.info(
            f"  Cluster {cluster_id}: {cell_type} ({confidence:.2f}) "
            f"— {n_cells} cells — {', '.join(top_markers[:5])}"
        )
    
    return cluster_annotations


def run_subcluster_pipeline(
    h5ad_dir: str | Path,
    metadata_path: str | Path,
    output_dir: str | Path,
    resolution: float = 0.8,
    resolution_mode: str = "adaptive",
    min_cluster_size: int = 20,
    signature_db_path: str = "",
) -> None:
    """Run full sub-cluster text generation: load metadata, process each h5ad, save JSONs."""
    h5ad_dir = Path(h5ad_dir)
    output_dir = Path(output_dir)

    with open(metadata_path) as f:
        metadata = json.load(f)

    extra_signatures = _load_signature_db(signature_db_path)
    signatures = {**CELL_TYPE_SIGNATURES, **extra_signatures}

    h5ad_files = sorted(h5ad_dir.glob("*_processed.h5ad"))
    all_subcluster_meta = {}
    total_clusters = 0
    total_annotated = 0
    unique_texts = set()

    for i, h5ad_path in enumerate(h5ad_files):
        dataset_id = h5ad_path.stem.replace("_processed", "")
        if dataset_id in metadata:
            meta = metadata[dataset_id]
            dataset_text = meta["text"] if isinstance(meta, dict) else meta
        else:
            dataset_text = f"Single-cell RNA sequencing data from {dataset_id}."

        try:
            adata = sc.read_h5ad(h5ad_path)
        except Exception as e:
            logger.warning(f"Failed to load {h5ad_path}: {e}")
            continue

        res = choose_resolution(int(adata.shape[0]), resolution, resolution_mode)
        cluster_annotations = cluster_and_annotate(
            adata, dataset_id, dataset_text,
            resolution=res,
            min_cluster_size=min_cluster_size,
            signatures=signatures,
        )

        if cluster_annotations:
            all_subcluster_meta[dataset_id] = {
                "dataset_text": dataset_text,
                "n_cells_total": int(adata.shape[0]),
                "n_clusters": len(cluster_annotations),
                "resolution": round(float(res), 3),
                "resolution_mode": resolution_mode,
                "clusters": cluster_annotations,
            }
            total_clusters += len(cluster_annotations)
            for cid, cinfo in cluster_annotations.items():
                if cinfo["confidence"] >= 0.25:
                    total_annotated += 1
                unique_texts.add(cinfo["text"])

    output_dir.mkdir(parents=True, exist_ok=True)
    subcluster_path = output_dir / "subcluster_metadata.json"
    with open(subcluster_path, "w") as f:
        json.dump(all_subcluster_meta, f, indent=2, ensure_ascii=True)

    expanded_metadata = {}
    for dataset_id, ds_info in all_subcluster_meta.items():
        for cluster_id, cluster_info in ds_info["clusters"].items():
            expanded_key = f"{dataset_id}__cluster_{cluster_id}"
            expanded_metadata[expanded_key] = {
                "text": cluster_info["text"],
                "cell_type": cluster_info["cell_type"],
                "confidence": cluster_info["confidence"],
                "n_cells": cluster_info["n_cells"],
                "source_dataset": dataset_id,
                "cluster_id": cluster_id,
                "top_markers": cluster_info["top_markers"],
            }

    expanded_path = output_dir / "metadata_subclusters.json"
    with open(expanded_path, "w") as f:
        json.dump(expanded_metadata, f, indent=2, ensure_ascii=True)

    logger.info(
        f"Sub-cluster analysis complete: {len(all_subcluster_meta)} datasets, "
        f"{total_clusters} clusters, {total_annotated} annotated (conf≥0.25), "
        f"{len(unique_texts)} unique texts. Saved to {subcluster_path} and {expanded_path}"
    )

