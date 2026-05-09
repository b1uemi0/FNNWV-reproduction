# Literature and Code Search Notes

## What was found

### FNNWV
- Found metadata for: FNNWV: farthest-nearest neighbor-based weighted voting for class-imbalanced crowdsourcing.
- DOI: https://doi.org/10.1007/s11432-023-3854-7
- Crossref metadata: references/fnnwv_paper/FNNWV_crossref.json
- Publisher/CDN PDF direct request was blocked by CloudWAF, so references/fnnwv_paper/FNNWV_s11432-023-3854-7.pdf is not a valid PDF.
- The user provided the paper locally. It was copied and extracted to:
  - references/fnnwv_paper/FNNWV_local.pdf
  - references/fnnwv_paper/FNNWV_local.txt

### IWMV
- Found open paper and downloaded PDF/text.
- Paper: Error Rate Bounds and Iterative Weighted Majority Voting for Crowdsourcing.
- arXiv: https://arxiv.org/abs/1411.4086
- Local files:
  - references/related_methods/IWMV_arxiv_1411.4086.pdf
  - references/related_methods/IWMV_arxiv_1411.4086.txt
- Public Java/CEKA implementation: not found in this search pass.

### MNLDP
- Found open IJCAI paper and downloaded PDF/text.
- Paper: Multiple Noisy Label Distribution Propagation for Crowdsourcing.
- IJCAI page: https://www.ijcai.org/proceedings/2019/204
- DOI: https://doi.org/10.24963/ijcai.2019/204
- Local files:
  - references/related_methods/MNLDP_IJCAI_2019_0204.pdf
  - references/related_methods/MNLDP_IJCAI_2019_0204.txt
- Important implementation detail from paper: k=5, eta=0.5, CEKA platform; edge weights solved by standard quadratic programming.
- Public Java/CEKA implementation: not found in this search pass.

### LAWMV
- Found article metadata/abstract but no open official PDF in this search pass.
- Paper: Label augmented and weighted majority voting for crowdsourcing.
- Venue: Information Sciences, Volume 606, 2022, Pages 397-409.
- DOI: https://doi.org/10.1016/j.ins.2022.05.066
- ScienceDirect page: https://www.sciencedirect.com/science/article/abs/pii/S0020025522004984
- Crossref metadata: references/related_methods/LAWMV_crossref.json
- Public Java/CEKA implementation: not found in this search pass.

## Code search status
Searches for class names and likely source file names did not find public code:
- "class IWMV" "ceka"
- "class LAWMV" "ceka"
- "class MNLDP" "ceka"
- "MNLDP.java"
- "MyQP" "MNLDP"

Current inference: these classes were probably author-side implementations used by the FNNWV authors, not included in CEKA 1.0.1 or this repository.
