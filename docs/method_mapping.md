# Method Mapping

This file maps the public SpaKD code to the terminology used in the manuscript.

| Manuscript term | Code symbol | Development-code origin |
| --- | --- | --- |
| scRNA-derived prediction branch | `student_encoder` | student branch |
| ST encoder branch | `teacher_encoder` | teacher branch |
| Shared structured basis-residual decoder | `StructuredBasisResidualDecoder` | `StructuredBasisDecoder` |
| Sample-level contrastive distillation (SCD) | `sample_contrastive_distillation_loss` | `scd_loss` |
| Global relational distillation (GRD) | `global_relational_distillation_loss` | `gram_align_loss` |
| Expression-derived prototype distillation (EPD) | `expression_prototype_distillation_loss` | `cpd_loss` |
| K-fold benchmark runner | `spakd.train_kfold` | `ensemble_kfold_cpd_fixed.py` |

The manuscript describes SCD, GRD, and EPD as the multi-level knowledge distillation objective. Extra exploratory terms from intermediate development scripts are not part of the public manuscript-aligned training objective.
