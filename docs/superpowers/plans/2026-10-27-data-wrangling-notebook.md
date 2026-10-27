# Data Wrangling Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a reproducible `main.ipynb` that completes Tasks 1–5 and exports the required long-format CSV.

**Architecture:** `wrangling_pipeline.py` will contain pure, testable pandas transformations. `main.ipynb` will use those functions in readable per-task cells, include tables, figures and Indonesian interpretations, and save `dataset long format.csv`. Pytest will check all rubric-critical invariants against the supplied Excel file.

**Tech Stack:** Python, pandas, numpy, matplotlib, seaborn, scikit-learn, pytest, nbformat, nbclient.

---

### Task 1: Establish data-table behaviour with tests

**Files:**
- Create: `tests/test_wrangling_pipeline.py`
- Create: `wrangling_pipeline.py`

- [ ] **Step 1: Write a failing table-splitting test.**

```python
def test_split_tables_creates_distinct_dimension_and_fact_tables(raw_data):
    dimension, facts = split_tables(raw_data)
    assert list(dimension.columns) == [
        "Kabupaten/Kota", "Provinsi", "Pulau", "Wilayah Indonesia", "Tipe Wilayah",
    ]
    assert dimension.shape == (514, 5)
    assert facts.shape == (2570, 25)
```

- [ ] **Step 2: Run `pytest -q tests/test_wrangling_pipeline.py`.**

Expected: failure because `split_tables` is not implemented.

- [ ] **Step 3: Implement `split_tables`.**

```python
DIMENSION_COLUMNS = ["Kabupaten/Kota", "Provinsi", "Pulau", "Wilayah Indonesia", "Tipe Wilayah"]

def split_tables(data):
    dimension = data.loc[:, DIMENSION_COLUMNS].drop_duplicates().reset_index(drop=True)
    facts = data.drop(columns=["Provinsi", "Pulau", "Wilayah Indonesia", "Tipe Wilayah"])
    return dimension, facts
```

- [ ] **Step 4: Re-run `pytest -q tests/test_wrangling_pipeline.py`.**

Expected: `1 passed`.

### Task 2: Add Task 2–4 transformations

**Files:**
- Modify: `tests/test_wrangling_pipeline.py`
- Modify: `wrangling_pipeline.py`

- [ ] **Step 1: Add a failing test for long data, panel data, merge cardinality, and stunting categories.**

```python
def test_task_two_to_four_transformations(raw_data):
    dimension, facts = split_tables(raw_data)
    long_data = make_long_data(raw_data)
    panel = make_stunting_panel(raw_data)
    merged = merge_dimension_and_facts(dimension, facts)
    enriched = add_stunting_category(raw_data)
    assert list(long_data.columns) == ["Kabupaten/Kota", "Tahun", "Nama Indikator", "Nilai"]
    assert long_data.shape == (59110, 4)
    assert list(panel.columns) == [2015, 2018, 2019, 2021, 2023]
    assert merged.shape[0] == 2570
    assert set(enriched.loc[enriched["Tahun"].eq(2023), "Kategori Stunting"]) <= {"Rendah", "Sedang", "Tinggi / Kritis"}
```

- [ ] **Step 2: Run `pytest -q tests/test_wrangling_pipeline.py`.**

Expected: failure names the absent transformation functions.

- [ ] **Step 3: Implement `make_long_data`, `make_stunting_panel`, `merge_dimension_and_facts`, hierarchical summary, per-island min-max normalisation, and `add_stunting_category`.**

```python
def make_long_data(data):
    id_columns = ["Kabupaten/Kota", "Tahun"]
    location_columns = ["Provinsi", "Pulau", "Wilayah Indonesia", "Tipe Wilayah"]
    indicators = [column for column in data.columns if column not in id_columns + location_columns]
    return data.melt(id_vars=id_columns, value_vars=indicators, var_name="Nama Indikator", value_name="Nilai")

def merge_dimension_and_facts(dimension, facts):
    return dimension.merge(facts, on="Kabupaten/Kota", how="left", validate="one_to_many")
```

- [ ] **Step 4: Re-run `pytest -q tests/test_wrangling_pipeline.py`.**

Expected: `2 passed`.

### Task 3: Add Task 5 analytical helpers

**Files:**
- Modify: `tests/test_wrangling_pipeline.py`
- Modify: `wrangling_pipeline.py`

- [ ] **Step 1: Add a failing test for IQR outliers and the requested correlation matrix.**

```python
def test_task_five_helpers(raw_data):
    outliers = find_iqr_outliers(raw_data)
    correlation = health_correlation(raw_data)
    assert outliers["Tahun"].eq(2023).all()
    assert {"PDRB per Kapita (Juta Rp)", "Kepadatan Penduduk (Jiwa/km2)"} <= set(outliers["Variabel"])
    assert correlation.shape == (5, 5)
```

- [ ] **Step 2: Run `pytest -q tests/test_wrangling_pipeline.py`.**

Expected: failure names the missing Task 5 functions.

- [ ] **Step 3: Implement `find_iqr_outliers` using 1.5×IQR separately for PDRB and density, and `health_correlation` with the five prescribed columns and Pearson correlation.**

```python
def find_iqr_outliers(data):
    subset = data.loc[data["Tahun"].eq(2023)].copy()
    records = []
    for column in ["PDRB per Kapita (Juta Rp)", "Kepadatan Penduduk (Jiwa/km2)"]:
        q1, q3 = subset[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        mask = subset[column].lt(q1 - 1.5 * iqr) | subset[column].gt(q3 + 1.5 * iqr)
        records.append(subset.loc[mask, ["Kabupaten/Kota", "Tahun", column]].rename(columns={column: "Nilai"}).assign(Variabel=column))
    return pd.concat(records, ignore_index=True)
```

- [ ] **Step 4: Re-run `pytest -q tests/test_wrangling_pipeline.py`.**

Expected: `3 passed`.

### Task 4: Build a clean executed notebook

**Files:**
- Modify: `main.ipynb`
- Create: `dataset long format.csv` (generated)
- Create: `heatmap_korelasi.png` (generated)
- Create: `boxplot_stunting_pulau.png` (generated)
- Create: `tren_stunting_wilayah.png` (generated)

- [ ] **Step 1: Replace notebook cells with headings for Task 1–5, source data checks, function calls, required outputs/figures, short Indonesian interpretations, and a conclusion.**

- [ ] **Step 2: Ensure the notebook writes `long_data.to_csv("dataset long format.csv", index=False)` and saves all three charts with the names above.**

- [ ] **Step 3: Execute from a clean kernel.**

Run: `python3 -m nbclient --execute --inplace --timeout 120 main.ipynb`

Expected: exit code 0 and current output in every code cell.

- [ ] **Step 4: Confirm generated artefacts.**

Run: `test -s 'dataset long format.csv' && test -s heatmap_korelasi.png && test -s boxplot_stunting_pulau.png && test -s tren_stunting_wilayah.png`

Expected: exit code 0.

### Task 5: Final verification and commit

**Files:**
- Verify: `tests/test_wrangling_pipeline.py`
- Verify: `main.ipynb`
- Verify: `dataset long format.csv`

- [ ] **Step 1: Run all transformation tests.**

Run: `pytest -q tests/test_wrangling_pipeline.py`

Expected: `3 passed`.

- [ ] **Step 2: Inspect all headings and generated files.**

Run: `rg -n 'Task [1-5]|Kesimpulan' main.ipynb && ls -l 'dataset long format.csv' heatmap_korelasi.png boxplot_stunting_pulau.png tren_stunting_wilayah.png`

Expected: the five task headings, conclusion, and four non-empty artefacts.

- [ ] **Step 3: Commit the verified implementation.**

```bash
git add main.ipynb wrangling_pipeline.py tests/test_wrangling_pipeline.py 'dataset long format.csv' heatmap_korelasi.png boxplot_stunting_pulau.png tren_stunting_wilayah.png
git commit -m "feat: complete data wrangling tasks"
```
