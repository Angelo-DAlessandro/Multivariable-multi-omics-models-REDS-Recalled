# Multivariable multi-omics models predict hemolysis and vesiculation phenotypes in stored red blood cells from the REDS-III RBC-Omics recalled donor cohort

This repository contains the Python workflow and input data used to generate the multivariable prediction atlas described in the manuscript:

**Multivariable multi-omics models predict hemolysis and vesiculation phenotypes in stored red blood cells from the REDS-III RBC-Omics recalled donor cohort**

The analysis evaluates whether donor-level metadata, CBC/ferritin variables, trace elements, metabolomics, lipidomics, and proteomics can predict stored red blood cell (RBC) hemolysis and extracellular vesicle (EV) phenotypes in the REDS-III RBC-Omics recalled donor cohort.

## Repository contents

| File | Description |
|---|---|
| `REDS_Recalled_Omics_predictors_Hemolysis.py` | Main Python script used to run the prediction workflow and generate figures, source tables, a review PDF, and a zipped output package. |
| `*.rar` / split archive parts | Compressed input table split into two `.rar` files because the complete table exceeded GitHub's single-file upload limit. Extract these parts together to recover the input CSV before running the script. |
| `Supplementary Methods and Figure.pdf` | Supplementary methods describing the modeling workflow and supplementary hive plot. |
| `Manuscript_v1_05182026.docx` | Manuscript draft associated with this analysis. |
| `README.md` | This file. |

## Study overview

The analysis focuses on donor-level day-42 measurements from the REDS-III RBC-Omics recalled donor cohort, which is enriched for donors at low and high extremes of hemolysis phenotypes. Depending on the endpoint, approximately 649 donors had sufficient predictor and outcome information for modeling.

Outcomes modeled independently include:

- spontaneous storage hemolysis
- osmotic hemolysis
- oxidative hemolysis
- total extracellular vesicles
- RBC extracellular vesicle subsets
- platelet extracellular vesicles
- change in total extracellular vesicles from storage day 10 to day 42

Predictor blocks include:

- metadata
- CBC/ferritin
- trace elements
- metabolomics
- lipidomics
- proteomics
- all-omics
- integrated full model

The central goal is to determine whether multivariable molecular scores can outperform single-feature association frameworks by aggregating many individually modest donor-level molecular signals into phenotype-specific predictive models.

## Requirements

The workflow was developed in Python and uses the following core packages:

```text
python >= 3.10
pandas
numpy
scipy
scikit-learn
matplotlib
```

The supplementary methods report the following implementation versions:

```text
Python 3.13.5
pandas 2.2.3
NumPy 2.3.5
scikit-learn 1.8.0
matplotlib 3.10.8
```

A typical setup is:

```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate      # Windows PowerShell

pip install pandas numpy scipy scikit-learn matplotlib
```

## Preparing the input table

The input table was uploaded as a split `.rar` archive because the full table exceeded GitHub's 25 MB single-file upload limit.

Download both `.rar` parts into the same folder, then extract the archive using WinRAR, 7-Zip, The Unarchiver, or an equivalent tool. Extraction should produce a CSV file equivalent to:

```text
Lead - REDS Recalled.csv
```

Do not rename or modify columns unless you also update the corresponding column names and column-index windows in the Python script.

## Running the analysis

From the repository directory, run:

```bash
python REDS_Recalled_Omics_predictors_Hemolysis.py \
  --input "Lead - REDS Recalled.csv" \
  --outdir "BloodAdvances_prediction_outputs"
```

On Windows PowerShell, use:

```powershell
python .\REDS_Recalled_Omics_predictors_Hemolysis.py `
  --input "Lead - REDS Recalled.csv" `
  --outdir "BloodAdvances_prediction_outputs"
```

The script expects a donor-level REDS recalled spreadsheet with the column structure used in the manuscript analysis.

## Workflow summary

For each outcome and predictor block, the script performs a leakage-safe 5-fold cross-validated ridge-regression analysis.

Within each training fold only, the workflow performs:

1. median imputation of missing numeric predictors
2. feature standardization
3. univariate feature screening
4. ridge-regression model fitting
5. prediction of the held-out fold

Held-out predictions from all five folds are then concatenated and summarized using cross-validated Pearson correlation and cross-validated R².

Feature stability is summarized from the cross-validation folds using:

- selection frequency
- mean absolute coefficient magnitude
- mean signed coefficient
- combined stability score

## Outputs

The script writes all outputs to the selected output directory.

Expected output files include:

| Output | Description |
|---|---|
| `prediction_performance_by_outcome_and_feature_block.csv` | Cross-validated performance for each outcome-by-feature-block model. |
| `best_prediction_model_per_outcome.csv` | Best-performing feature block for each modeled phenotype. |
| `model_feature_importance_stability.csv` | Stable predictor summaries across folds. |
| `top25_full_model_features_per_outcome.csv` | Top stable full-model predictors per outcome. |
| `Figure_1_BA_prediction_workflow_performance.png` | Main Figure 1 PNG. |
| `Figure_1_BA_prediction_workflow_performance_text.svg` | Main Figure 1 SVG with editable text where possible. |
| `Figure_2_BA_prediction_accuracy_hive.png` | Main Figure 2 PNG. |
| `Figure_2_BA_prediction_accuracy_hive_text.svg` | Main Figure 2 SVG with editable text where possible. |
| `Figure_3_BA_predictor_architecture.png` | Main Figure 3 PNG. |
| `Figure_3_BA_predictor_architecture_text.svg` | Main Figure 3 SVG with editable text where possible. |
| `Supplementary_Figure_1_fullpage_hive.png` | Full-page supplementary hive plot. |
| `BloodAdvances_prediction_figures_review.pdf` | Review PDF containing the generated figures. |
| `BloodAdvances_prediction_outputs_package.zip` | Zipped package of generated outputs. |

## Reproducibility notes

All preprocessing, feature screening, and model fitting are performed inside the training folds before application to the held-out fold. This design prevents information from the held-out test fold from entering imputation, scaling, feature selection, or model training.

The analysis uses ridge regression because the omics predictors are high-dimensional and correlated. The workflow is intended as a discovery and benchmarking framework for phenotype-specific donor-level prediction, not as a final locked clinical model.

## Important caveats

- The recalled cohort is enriched for donors at hemolysis extremes and is therefore not a population-random sample.
- The input table must be extracted from both `.rar` parts before running the script.
- The `--skip-modeling` flag is present in the script but currently raises an error because observed-versus-predicted values are not serialized for later figure regeneration. Run the script without `--skip-modeling`.
- External validation in the full index cohort and prospective testing in independent donor populations are needed before clinical deployment.

## Citation

If using this code or derived outputs, please cite the associated manuscript:

Dzieciatkowska M, Stephenson D, Nemkov T, Stone M, Kleinman S, Busch MP, Norris PJ, D'Alessandro A. **Multivariable multi-omics models predict hemolysis and vesiculation phenotypes in stored red blood cells from the REDS-III RBC-Omics recalled donor cohort.**

## Contact

Angelo D'Alessandro, PhD  
Department of Biochemistry and Molecular Genetics  
University of Colorado Anschutz Medical Campus  
Email: angelo.dalessandro@cuanschutz.edu
