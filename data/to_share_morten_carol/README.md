CEDAR : Neoepitope dump (June 2022) from Zeynep, filtered to keep only peptides with wild_type
PRIME : From PRIME's suppl material, filtered to keep only neoepitope (removing viral+random peptides), and keeping 
only those that have HLA in common with CEDAR / exists in the IC dictionaries

Columns of interest:

- Peptide : full mutant peptide sequence
- wild_type : full wt peptide sequence
- HLA : HLA with full resolution
- agg_label : target value (immunogenicity, 0 or 1)
- EL_rank_mut : Rank of mutant (found using the ".fa" input method)
- EL_rank_wt : Rank of WT (same)
- core_mut, core_wt, icore_mut, icore_wt, Peptide_mut, Peptide_wt : Peptide, Core and Icore found using NetMHCpan with 
.fa as input 
- mhc_anchor : Anchor position of a given HLA allele associated with the given peptide, taken from IC content of given 
HLA considering length 9, (found with a threshold of 0.1615, chose this value because it's the minimum value at which 
most alleles have comparable/"correct" anchor position when comparing against the logos found on NetMHCpan's webserver 
motif viewer)
- anchor_mutation : Whether a given position is located on the anchor (Using the core to compare as opposed to full 
peptide)
- binder : Whether a given mutant is an Improved or Conserved binder
- mutation_score, blsm_mut_score, core_mutation_score, core_blsm_mut_score : Mutation & BLOSUM mutation scores found 
either using the full peptide or the core
- ratio_rank : rank_wt / rank_mut
- dissimilarity_score : 1 - kernel_similarity(mutant,wild_type)
- aliphatic_index, boman, hydrophobicity, isoelectric_point, VHSE1, VHSE3, VHSE7, VHSE8 : amino acid properties 
(aa_props used as features), computed using the full peptide
- diff_label_from_prime : ONLY for CEDAR, peptide for which the labels differ from PRIME

