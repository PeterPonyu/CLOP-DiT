#!/usr/bin/env python3
"""
Text Polishing Script for CLOP-DiT
==================================

Transforms 1088 template texts from rigid "In {org} {tissue} ({disease}): {prefix} {cell_type} 
identified by expression of {markers}" format into semantically rich, natural language descriptions 
that emphasize cell-type biology over tissue/dataset context.

Strategy:
1. Extract structured fields (organism, tissue, disease, cell_type, markers) from templates
2. Enrich each cell type with biological function/signature knowledge
3. Generate natural language descriptions emphasizing cell identity + function + markers
4. Preserve dataset/tissue context but demote it below cell-type semantics
5. Save polished texts to new JSON and update cache

Output:
- data/cached_latents_v5.2/text_strings_polished.json (before re-embedding)
- data/cached_latents_v5.2/text_strings.json (replaced with polished version if successful)
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple
import sys

# Biological knowledge base: cell type → (biological_function, markers_interpretation)
CELL_TYPE_KNOWLEDGE = {
    # Immune cells
    'CD8+ cytotoxic T lymphocytes': (
        'Effector T cells with cytotoxic capacity, critical for anti-viral and anti-tumor immunity',
        'express T cell co-receptors (CD3, CD8) and cytotoxic granule components (GZMA, GZMB, PRF1, NKG7)'
    ),
    'CD4+ helper T lymphocytes': (
        'Helper T cells coordinating adaptive immune responses through cytokine secretion and antigen presentation',
        'express CD3, CD4, and lymphocyte trafficking molecules (CCR7, SELL) or T-box transcription factors (TCF7)'
    ),
    'CD4+FOXP3+ regulatory T cells (Tregs)': (
        'Immunosuppressive T cells critical for immune tolerance and prevention of autoimmunity',
        'express CD4, CD25/IL2RA, FOXP3, and immune checkpoint molecules'
    ),
    'B lymphocytes': (
        'Lymphocytes responsible for antibody production and humoral immunity',
        'express B cell markers (CD19, CD20) and may show light chain restriction'
    ),
    'natural killer cells': (
        'Innate lymphocytes capable of direct cytotoxic killing and IFN-γ production without prior sensitization',
        'express NK receptors (NKG7, GNLY, GZMB) and activating molecules'
    ),
    'gamma-delta T lymphocytes': (
        'T lymphocytes using γδ TCR for antigen recognition, important in mucosal and epithelial immunity',
        'express T cell markers (CD3) with γδ TCR chain genes'
    ),
    'antibody-secreting plasma cells': (
        'Differentiated B cells actively secreting large quantities of specific antibodies',
        'express immunoglobulin heavy and light chain genes (IGHA1, IGHG1, IGHE) and plasma cell markers (SDC1, CD38)'
    ),
    'MHC-II-high antigen-presenting cells': (
        'Professional antigen-presenting cells including dendritic cells and activated macrophages with high MHC-II expression',
        'express CD74 and HLA-DQ/DP/DR molecules for MHC-II-mediated antigen presentation'
    ),
    'dendritic cells': (
        'Specialized antigen-presenting cells bridging innate and adaptive immunity',
        'express DC markers and co-stimulatory molecules'
    ),
    'plasmacytoid dendritic cells': (
        'Subset of dendritic cells specialized in type I interferon production during viral infection',
        'express pDC-specific markers and interferon response genes'
    ),
    
    # Myeloid cells
    'circulating monocytes': (
        'Circulating myeloid cells serving as precursors to tissue macrophages and dendritic cells',
        'express CD14, CST3, FCN1, and other monocyte-specific genes'
    ),
    'tissue-resident macrophages': (
        'Long-lived innate immune cells providing tissue homeostasis and pathogen surveillance',
        'express macrophage markers (CD68, APOE) and may be specialized for particular tissues'
    ),
    'alveolar macrophages': (
        'Tissue-resident macrophages in the lung specialized for surveillance of respiratory surfaces',
        'express alveolar macrophage markers and surfactant protein recognition genes'
    ),
    'liver Kupffer cells (tissue-resident macrophages)': (
        'Specialized tissue-resident macrophages in the liver involved in immune surveillance and waste clearance',
        'express macrophage and liver-homing markers'
    ),
    'microglia': (
        'Specialized macrophages of the central nervous system surveying brain tissue and removing debris',
        'express microglia-specific markers and CNS tissue signatures'
    ),
    'neutrophils': (
        'Most abundant white blood cells, first responders to infection with antimicrobial granules',
        'express S100A8, S100A9, CSF3R, FCGR3B and other neutrophil differentiation markers'
    ),
    'mast cells': (
        'Tissue-resident immune cells involved in allergic reactions and parasitic infection responses',
        'express mast cell granule proteases and high-affinity IgE receptors'
    ),
    'exhausted T cells with chronic activation markers': (
        'T cells unable to mount effective responses due to sustained antigen exposure in chronic conditions',
        'express exhaustion-associated checkpoint molecules and reduced effector function'
    ),
    'interferon-stimulated / antiviral response cells': (
        'Cells with activated type I/III interferon responses mounted against viral infection',
        'express ISG15, OAS1, MX1, IFIT1 and other interferon-stimulated genes'
    ),
    'cells in heat-shock / stress response state': (
        'Cells experiencing proteotoxic or other cellular stress with activated heat shock protein responses',
        'express HSP90AA1, HSPA1A, HSPD1, DNAJB1 and other molecular chaperones'
    ),

    # Stromal and structural cells
    'fibroblasts / mesenchymal stromal cells': (
        'Non-contractile mesenchymal cells providing structural support and producing extracellular matrix',
        'express FN1, COL1A1, COL3A1, and stromal markers (FSP1, αSMA in some contexts)', 
    ),
    'smooth muscle cells': (
        'Contractile cells responsible for vasoregulation and other tissue contractions',
        'express smooth muscle markers including contractile proteins (ACTA2, SMS22A)'
    ),
    'pericytes / mural cells': (
        'Mesenchymal cells surrounding blood vessels providing structural support and vascular stability',
        'express pericyte markers and produce stabilizing factors'
    ),
    'myoepithelial / basal cells': (
        'Contractile epithelial cells providing mechanical force and support at epithelial-stromal interfaces',
        'express both epithelial and contractile markers'
    ),
    'vascular endothelial cells': (
        'Cells lining blood vessels forming the barrier between blood and tissues',
        'express endothelial markers (CDH5, PECAM1, VWF, EMCN)'
    ),
    'venous endothelial cells': (
        'Specialized endothelial cells from venous vasculature with distinct transcriptomic signatures',
        'express venous-specific markers'
    ),
    'capillary endothelial cells': (
        'Specialized endothelial cells from capillary networks with tight junctions and high permeability selectivity',
        'express capillary-specific markers'
    ),
    'tip / angiogenic endothelial cells': (
        'Leading-edge endothelial cells during angiogenesis with migration and sprouting capacity',
        'express angiogenic and migration-associated genes'
    ),
    'lymphatic endothelial cells': (
        'Specialized endothelial cells forming lymphatic vessels, structurally and functionally distinct from blood endothelium',
        'express lymphatic-specific molecules (PROX1, LYVE1, PODOPLANIN)'
    ),

    # Epithelial cells - general
    'epithelial cells': (
        'Specialized cells forming protective barriers in tissues',
        'express epithelial cell markers'
    ),
    'basal epithelial cells': (
        'Progenitor epithelial cells in basal layers with regenerative capacity',
        'express basal epithelial markers and remain attached to basement membrane'
    ),
    'luminal epithelial cells': (
        'Specialized epithelial cells forming the luminal surface of ducts and organs',
        'express luminal epithelial markers'
    ),
    'multiciliated epithelial cells': (
        'Epithelial cells bearing multiple cilia for fluid clearance and transport',
        'express cilial genes and CCDC genes'
    ),

    # Respiratory epithelium
    'alveolar type 1 pneumocytes': (
        'Thin gas exchange cells covering majority of alveolar surface in lungs',
        'express AT1-specific markers (AGER, AQP5)'
    ),
    'alveolar type 2 pneumocytes': (
        'Cuboidal alveolar cells producing pulmonary surfactant and serving as alveolar stem cells',
        'express surfactant proteins (SFTPA1, SFTPB, SFTPC, SFTPD)'
    ),
    'club (Clara) secretory cells': (
        'Secretory epithelial cells in airways producing surfactant and antimicrobial proteins',
        'express club cell secretory proteins (SCGB1A1)'
    ),

    # Digestive epithelium
    'hepatocytes': (
        'Major metabolic cells of the liver performing detoxification and synthesis',
        'express liver-specific metabolic enzymes and albumin'
    ),
    'cholangiocytes / bile duct epithelial cells': (
        'Epithelial cells lining bile ducts involved in bile transport and modification',
        'express bile duct markers and hepatic transporters'
    ),
    'mucus-secreting goblet cells': (
        'Intestinal epithelial cells secreting protective mucus layers',
        'express mucin genes (MUC2, MUC5AC)'
    ),
    'kidney proximal tubule cells': (
        'Epithelial cells of renal proximal tubules specialized in reabsorption and secretion',
        'express proximal tubule-specific transporters and metabolic enzymes'
    ),
    'kidney collecting duct cells': (
        'Epithelial cells of renal collecting duct regulating water and electrolyte balance',
        'express aquaporins and ion channels'
    ),
    'glomerular podocytes': (
        'Specialized epithelial cells of glomerular filtration barrier with intricate foot processes',
        'express nephrin, podocin, and other slit diaphragm proteins'
    ),

    # Reproductive/endocrine epithelium
    'prostate epithelial cells': (
        'Specialized epithelial cells of the prostate producing secretory proteins',
        'express prostate-specific antigens and secretory proteins'
    ),
    'Sertoli cells (testicular supporting cells)': (
        'Specialized epithelial cells supporting spermatogenesis and producing anti-Müllerian hormone',
        'express Sertoli cell markers'
    ),
    'mammary secretory epithelial cells': (
        'Epithelial cells of mammary glands producing milk proteins',
        'express milk protein genes'
    ),

    # Pancreatic endocrine cells
    'pancreatic alpha cells': (
        'Endocrine cells producing glucagon for glucose mobilization',
        'express GCG and glucagon-regulating genes'
    ),
    'pancreatic beta cells': (
        'Endocrine cells producing insulin for glucose homeostasis',
        'express INS and β-cell-specific transcription factors (PDX1, NEUROD1)'
    ),
    'pancreatic delta cells': (
        'Endocrine cells producing somatostatin for regulation of neighboring endocrine cells',
        'express SST and delta cell markers'
    ),

    # Pituitary cells
    'pituitary corticotroph cells (ACTH-producing)': (
        'Endocrine cells producing adrenocorticotropic hormone regulating stress response',
        'express POMC'
    ),
    'pituitary gonadotroph cells (FSH/LH-producing)': (
        'Endocrine cells producing follicle-stimulating hormone and luteinizing hormone for reproductive function',
        'express FSHB and LHB'
    ),
    'pituitary somatotroph cells (growth hormone-producing)': (
        'Endocrine cells producing growth hormone regulating growth and metabolism',
        'express GH1'
    ),
    'pituitary thyrotroph cells (TSH-producing)': (
        'Endocrine cells producing thyroid-stimulating hormone regulating thyroid function',
        'express TSHB'
    ),

    # Dermal/epidermal cells
    'melanocytes': (
        'Pigment-producing cells of skin providing photoprotection',
        'express melanin synthesis genes (TYROSINASE, PMEL, DCT)'
    ),
    'Merkel cells (neuroendocrine touch receptors)': (
        'Mechanosensory neuroendocrine cells in skin detecting light touch',
        'express touch transduction genes and neuroendocrine markers'
    ),

    # Dental cells
    'ameloblasts (enamel-producing cells)': (
        'Specialized secretory cells producing tooth enamel matrix proteins',
        'express amelogenin and other enamel matrix proteins'
    ),
    'odontoblasts / dentin-secreting cells': (
        'Specialized secretory cells producing dentin matrix and involved in tooth innervation',
        'express dentin matrix genes (DSPP, DMP1)'
    ),
    'dental epithelial progenitor cells': (
        'Progenitor cells of dental epithelium with regenerative capacity',
        'express dental epithelial progenitor markers'
    ),

    # Neural cells - neurons
    'neurons': (
        'Electrically excitable cells transmitting signals through action potentials and synaptic transmission',
        'express neuronal markers (MAP2, TUBB3, SYNAPTOTAGMIN1)'
    ),
    'excitatory (glutamatergic) neurons': (
        'Neurons using glutamate as excitatory neurotransmitter',
        'express glutamatergic markers'
    ),
    'inhibitory (GABAergic) neurons': (
        'Neurons using GABA as inhibitory neurotransmitter',
        'express GABAergic markers'
    ),
    'motor neurons': (
        'Neurons projecting to muscles and controlling voluntary movement',
        'express motor neuron markers'
    ),
    'dorsal horn sensory neurons': (
        'Neurons of spinal cord dorsal horn processing sensory information',
        'express sensory neuron markers'
    ),
    'spinal cord interneurons': (
        'Neurons integrating sensory and motor signals within spinal cord',
        'express interneuron markers'
    ),
    'hippocampal dentate gyrus granule cells': (
        'Neurons of hippocampus involved in memory formation and pattern separation',
        'express hippocampal granule cell markers'
    ),
    'retinal ganglion cells': (
        'Output neurons of retina projecting to brain visual centers',
        'express retinal ganglion cell markers'
    ),
    'retinal bipolar cells': (
        'Interneurons of retina relaying photoreceptor signals to ganglion cells',
        'express bipolar cell markers'
    ),

    # Neural cells - glia
    'astrocytes': (
        'Support cells providing metabolic, structural, and synaptic support to neurons',
        'express astrocyte markers (GFAP, ALDH1L1, SOX9)'
    ),
    'oligodendrocytes': (
        'Myelinating glia producing myelin sheaths insulating axons in CNS',
        'express oligodendrocyte markers (MOBP, MOG)'
    ),
    'oligodendrocyte precursor cells (OPCs)': (
        'Progenitor cells with capacity to differentiate into oligodendrocytes',
        'express OPC markers (NG2/CSPG4, PDGFRA)'
    ),
    'Müller glial cells': (
        'Support cells of retina providing metabolic support and structural scaffold',
        'express Müller cell markers'
    ),
    'Schwann cells (peripheral glia)': (
        'Myelinating glia of peripheral nervous system producing myelin around axons',
        'express Schwann cell markers (S100B, PMP22)'
    ),
    'radial glial / neural progenitor cells': (
        'Progenitor cells of developing nervous system with radial processes',
        'express radial glial markers (APOD, BLBP)'
    ),
    'intermediate neural progenitors': (
        'Intermediate progenitor cells with reduced self-renewal and increased neuronal commitment',
        'express intermediate progenitor markers'
    ),

    # Cell cycle and proliferation
    'actively cycling cells (S/G2/M phase)': (
        'Cells actively progressing through DNA synthesis and mitosis phases of cell cycle',
        'express S phase markers (TUBA1B, TUBB, TOP2A) and mitotic markers (AURKB, CENPE)'
    ),
    'actively proliferating cells': (
        'Cells with high proliferative activity and rapid division rates',
        'express proliferation markers (Ki67/MKI67, PCNA)'
    ),

    # Stem and progenitor cells
    'hematopoietic stem and progenitor cells': (
        'Self-renewing blood-forming progenitors with multilineage differentiation potential',
        'express HSC markers (CD34, CD38 status)'
    ),
    'pluripotent stem cells (ESC/iPSC)': (
        'Undifferentiated cells with unlimited self-renewal and capacity to form all cell types',
        'express pluripotency factors (OCT4, SOX2, NANOG)'
    ),

    # Blood cells
    'erythroid lineage cells': (
        'Cells of erythrocyte lineage at various developmental stages',
        'express erythroid markers (GYPA, HBA1, HBA2, HBB)'
    ),
    'mature red blood cells / erythrocytes': (
        'Fully mature oxygen-carrying cells of blood lacking nucleus',
        'express hemoglobin genes (HBA1, HBA2, HBB)'
    ),
    'megakaryocytes / platelet-producing cells': (
        'Large bone marrow cells producing platelets for hemostasis',
        'express megakaryocyte markers (PF4, PPBP)'
    ),

    # Other cell types
    'uncharacterized cells': (
        'Cell population with unresolved identity, possibly representing novel or rare types',
        'expressed marker genes'
    ),
    'mesenchymal stem / stromal cells': (
        'Multipotent progenitor cells with capacity to differentiate into various mesenchymal lineages',
        'express mesenchymal stem cell markers'
    ),
    'neuroendocrine cells': (
        'Cells combining neuronal and endocrine functions',
        'express neuroendocrine markers'
    ),
    'definitive endoderm cells': (
        'Embryonic cells of definitive endoderm germ layer',
        'express endodermal markers'
    ),
    'mesothelial cells': (
        'Epithelial cells covering serosal cavities (peritoneum, pleura, pericardium)',
        'express mesothelial markers'
    ),
    'tuft (chemosensory) cells': (
        'Rare epithelial cells with chemosensory function and type 2 immune signaling',
        'express tuft cell markers'
    ),
    'adipocytes': (
        'Lipid storage cells with energy metabolism functions',
        'express adipocyte markers (PPARγ, FABP4)'
    ),
    'urothelial / transitional epithelial cells': (
        'Specialized epithelial cells of bladder with stretch accommodation capacity',
        'express uroplakin genes'
    ),
}

def extract_template_fields(text: str) -> Optional[Dict]:
    """Parse template text into structured fields."""
    # Pattern: In {organism} {tissue} ({disease}): {prefix} {cell_type} identified by expression of {markers}
    m = re.match(r'^In (.+?):\s*(.+)$', text)
    if not m:
        return None
    
    context = m.group(1).strip()
    body = m.group(2).strip()
    
    # Parse organism, tissue, disease
    ctx_m = re.match(r'(.+?)\s+(.+?)(?:\s+\((.+?)\))?$', context)
    if not ctx_m:
        return None
    organism = ctx_m.group(1)
    tissue = ctx_m.group(2)
    disease = ctx_m.group(3) or 'normal'
    
    # Parse prefix, cell_type, markers
    ct_m = re.match(r'(Sub-population of|Putative|Candidate)\s+(.+?)\s+identified by expression of\s+(.+?)\.?$', body)
    if ct_m:
        prefix = ct_m.group(1)
        cell_type = ct_m.group(2)
        markers = ct_m.group(3)
    elif 'Uncharacterized cell population' in body:
        uc_m = re.match(r'Uncharacterized cell population with high expression of\s+(.+?)\.?$', body)
        if uc_m:
            prefix = 'Uncharacterized'
            cell_type = 'uncharacterized cells'
            markers = uc_m.group(1)
        else:
            return None
    else:
        return None
    
    return {
        'organism': organism,
        'tissue': tissue,
        'disease': disease,
        'prefix': prefix,
        'cell_type': cell_type,
        'markers': markers,
    }

def format_markers(marker_str: str) -> str:
    """Format marker genes into readable list."""
    markers = [m.strip() for m in marker_str.split(',')]
    if len(markers) <= 3:
        return ' and '.join(markers)
    elif len(markers) <= 6:
        return ', '.join(markers[:-1]) + ' and ' + markers[-1]
    else:
        return ', '.join(markers[:4]) + f' and {len(markers)-4} other markers'

def format_tissue_context(organism: str, tissue: str, disease: str) -> str:
    """Format tissue context naturally."""
    if disease != 'normal':
        return f"in {organism} {tissue} affected by {disease}"
    else:
        return f"in {organism} {tissue}"

def polish_template_text(text: str, text_id: str) -> str:
    """Transform template text into semantically rich description."""
    fields = extract_template_fields(text)
    if not fields:
        # Not a template - return as-is (dataset-level descriptions)
        return text
    
    cell_type = fields['cell_type']
    markers_formatted = format_markers(fields['markers'])
    tissue_context = format_tissue_context(fields['organism'], fields['tissue'], fields['disease'])
    
    # Get biological knowledge if available
    if cell_type in CELL_TYPE_KNOWLEDGE:
        function, marker_interpretation = CELL_TYPE_KNOWLEDGE[cell_type]
        
        # Construct polished text: cell identity + function + markers in natural language
        polished = (
            f"{cell_type.capitalize()} are {function}. "
            f"In this dataset, these cells were identified {tissue_context} by expression of "
            f"{markers_formatted}. {marker_interpretation.capitalize() if marker_interpretation else 'These marker genes are characteristic of this cell type.'}"
        )
    else:
        # Fallback for unknown cell types
        polished = (
            f"{cell_type.capitalize()} were identified {tissue_context} by expression of "
            f"{markers_formatted}."
        )
    
    return polished

def process_all_texts(input_path: Path, output_path: Path) -> Tuple[Dict, int, int]:
    """Polish all texts and save to output."""
    print(f"Loading texts from {input_path}...")
    with open(input_path) as f:
        texts = json.load(f)
    
    polished = {}
    num_template = 0
    num_preserved = 0
    
    print(f"\nPolishing {len(texts)} texts...")
    for idx, (text_id, text) in enumerate(texts.items()):
        polished_text = polish_template_text(text, text_id)
        polished[text_id] = polished_text
        
        # Track changes
        if polished_text != text:
            num_template += 1
        else:
            num_preserved += 1
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(texts)}")
    
    print(f"\nSaving polished texts to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(polished, f, indent=2)
    
    return polished, num_template, num_preserved

def main():
    cache_dir = Path("/home/zeyufu/Desktop/CLOP-DiT/data/cached_latents_v5.2")
    input_file = cache_dir / "text_strings.json"
    output_file = cache_dir / "text_strings_polished.json"
    
    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        sys.exit(1)
    
    print("="*80)
    print("CLOP-DiT Text Polishing Tool")
    print("="*80)
    
    polished, num_template, num_preserved = process_all_texts(input_file, output_file)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total texts processed: {len(polished)}")
    print(f"Template texts polished: {num_template}")
    print(f"Dataset-level texts preserved: {num_preserved}")
    print(f"Output saved to: {output_file}")
    print("\nSample polished texts:")
    print("-"*80)
    
    # Show samples
    samples = list(polished.items())
    for text_id, text in samples[:3]:
        print(f"\nID {text_id}:")
        print(f"  {text[:200]}...")
    
    print("\n" + "="*80)
    print("Next steps:")
    print("1. Review sample polished texts above for quality")
    print("2. If satisfied, run: mv text_strings_polished.json text_strings.json")
    print("3. Re-embed with BiomedBERT and update cache")
    print("="*80)

if __name__ == "__main__":
    main()
