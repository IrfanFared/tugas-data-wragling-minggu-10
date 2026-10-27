# Data Wrangling Notebook Design

## Goal

Replace `main.ipynb` with a reproducible, presentation-ready notebook that completes Task 1 through Task 5 in `Tugas Data_Wrangling sebelum UTS.pdf`, and exports the required long-format CSV.

## Inputs and outputs

- Input: `Dataset_Kesehatan_Real_Names.xlsx`.
- Primary notebook: `main.ipynb`.
- Required generated artefact: `dataset long format.csv`.
- Validation support: a small automated test module that independently checks the required table shapes, transformations, merge cardinality, category rules, and generated artefacts.

## Notebook flow

1. Load the Excel dataset and show its structure, duplicate count, and missing values.
2. Create a distinct five-column region dimension table and a 25-column health-fact table. Display shape, dtypes, and missing values for both.
3. Melt the 23 indicators to a four-column tidy table, rename its fields to `Nama Indikator` and `Nilai`, save it as `dataset long format.csv`, and build the requested stunting panel.
4. Filter the 2023 stunting-and-poverty condition, then perform a left join using the distinct dimension table to preserve the 2,570 fact rows.
5. Calculate hierarchical AKI and stunting summaries, perform within-island min-max normalisation, and add a 2023 `Kategori Stunting` field using the supplied thresholds.
6. Detect 2023 outliers with IQR, produce the requested Pearson heatmap, boxplot, and regional line plot.
7. Add concise Indonesian interpretations after every task and a final conclusion based on the generated results.

## Reliability and validation

The notebook runs from a clean kernel without relying on stale outputs. Tests assert: region-dimension shape is 514 by 5, fact-table shape is 2,570 by 25, tidy data has 59,110 rows and the prescribed column names, the left join has 2,570 rows, stunting labels observe the boundary rules, and the CSV plus three figures are generated.

## Scope boundaries

The work updates the notebook and creates the required CSV and test support. It does not create the separate formal report PDF because that requires a report format and narrative beyond the requested notebook correction.
