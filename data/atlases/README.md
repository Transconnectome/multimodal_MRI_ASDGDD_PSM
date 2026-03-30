# Atlas and Template Files

Pre-computed atlas and template files required for the XAI tract overlap analysis (Step 7).

## Files

| File | Shape | Voxel Size | Description |
|------|-------|-----------|-------------|
| `alltracts.nii.gz` | 195 x 233 x 159 x 87 | 1mm iso | Probabilistic white matter tract atlas (87 tracts) in infant MNI space |
| `hcp1065_abbreviation.txt` | - | - | Tract index, abbreviation, and full name (UTF-16LE encoded) |
| `infant_MNI_template.nii.gz` | 195 x 233 x 159 | 1mm iso | Infant MNI structural template (33-44 months age range) |

## HCP 1065 White Matter Tractography Atlas

87 white matter tracts reconstructed from 1,065 Human Connectome Project subjects using deterministic fiber tracking in DSI Studio.

- **Original atlas**: Adult MNI space (ICBM152), warped to infant MNI for this study
- **Method**: Generalized q-sampling imaging (GQI) + deterministic tracking
- **Tracts**: 87 bundles including major association (SLF, IFOF, UF, AF), projection (CST, CPT, OR), commissural (CC, AC), limbic (CB, FX), and cerebellar (SCP, MCP, ICP) pathways

**References:**
- Yeh FC, Panesar S, Fernandes D, et al. (2018). Population-averaged atlas of the macroscale human structural connectome and its network topology. *NeuroImage*, 178, 57-68. https://doi.org/10.1016/j.neuroimage.2018.05.027
- Yeh FC, Verstynen TD, Wang Y, et al. (2013). Deterministic diffusion fiber tracking improved by quantitative anisotropy. *PLoS ONE*, 8(11), e80713. https://doi.org/10.1371/journal.pone.0080713

**Software:**
- DSI Studio: https://dsi-studio.labsolver.org/
- Atlas download: https://brain.labsolver.org/hcp_trk_atlas.html
- Tract documentation: https://dsi-studio.labsolver.org/doc/gui_t3_whole_brain.html

## Infant MNI Template

Age-specific brain template for infant populations, used as the registration target for group-level saliency analysis.

- **Age range**: 33-44 months (matching the study cohort mean age)
- **Resolution**: 1mm isotropic, 195 x 233 x 159 voxels
- **Construction**: Nonlinear groupwise registration of infant structural MRIs

**References:**
- Shi F, Yap PT, Wu G, et al. (2011). Infant brain atlases from neonates to 1- and 2-year-olds. *PLoS ONE*, 6(4), e18746. https://doi.org/10.1371/journal.pone.0018746
- Fonov V, Evans AC, Botteron K, et al. (2011). Unbiased average age-appropriate atlases for pediatric studies. *NeuroImage*, 54(1), 313-327. https://doi.org/10.1016/j.neuroimage.2010.07.033

**Download:**
- UNC Infant Brain Atlases: https://www.nitrc.org/projects/pediatricatlas
- NIH Pediatric MRI Data Repository: https://pediatricmri.nih.gov/

## Notes

- All three files are co-registered to the same infant MNI space and can be directly overlaid.
- The tract atlas contains probabilistic masks (float, 0-1 range) per tract.
- `alltracts.nii.gz` was derived by warping the adult HCP 1065 atlas to the infant template using ANTs SyN registration.
