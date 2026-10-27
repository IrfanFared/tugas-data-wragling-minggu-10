from pathlib import Path

import pandas as pd

import wrangling_pipeline as pipeline

from wrangling_pipeline import split_tables


def test_split_tables_creates_distinct_dimension_and_fact_tables():
    raw_data = pd.read_excel(Path(__file__).parents[1] / "Dataset_Kesehatan_Real_Names.xlsx")
    expected_facts = raw_data.drop(
        columns=["Provinsi", "Pulau", "Wilayah Indonesia", "Tipe Wilayah"]
    )

    dimension, facts = split_tables(raw_data)

    assert list(dimension.columns) == [
        "Kabupaten/Kota",
        "Provinsi",
        "Pulau",
        "Wilayah Indonesia",
        "Tipe Wilayah",
    ]
    assert dimension.shape == (514, 5)
    assert not dimension.duplicated().any()
    assert facts.shape == (2570, 25)
    assert list(facts.columns) == list(expected_facts.columns)
    pd.testing.assert_frame_equal(facts, expected_facts)


def test_long_data_has_required_tidy_columns_and_all_indicator_values():
    raw_data = pd.read_excel(Path(__file__).parents[1] / "Dataset_Kesehatan_Real_Names.xlsx")

    long_data = pipeline.make_long_data(raw_data)

    assert list(long_data.columns) == ["Kabupaten/Kota", "Tahun", "Nama Indikator", "Nilai"]
    assert long_data.shape == (59110, 4)
    assert long_data["Nama Indikator"].nunique() == 23


def test_stunting_panel_has_requested_year_columns():
    raw_data = pd.read_excel(Path(__file__).parents[1] / "Dataset_Kesehatan_Real_Names.xlsx")

    panel = pipeline.make_stunting_panel(raw_data)

    assert list(panel.columns) == [2015, 2018, 2019, 2021, 2023]
    assert panel.shape == (514, 5)


def test_2023_filter_and_left_join_preserve_expected_rows():
    raw_data = pd.read_excel(Path(__file__).parents[1] / "Dataset_Kesehatan_Real_Names.xlsx")
    dimension, facts = split_tables(raw_data)

    filtered = pipeline.filter_stunting_and_poverty_2023(raw_data)
    merged = pipeline.merge_dimension_and_facts(dimension, facts)

    assert filtered["Tahun"].eq(2023).all()
    assert filtered["Prevalensi Stunting (%)"].gt(0.30).all()
    assert filtered["Penduduk Miskin (%)"].gt(0.10).all()
    assert merged.shape == (2570, 29)


def test_task_four_summaries_normalization_and_categories():
    raw_data = pd.read_excel(Path(__file__).parents[1] / "Dataset_Kesehatan_Real_Names.xlsx")

    summary = pipeline.summarize_health_by_region(raw_data)
    normalized = pipeline.minmax_by_island(raw_data)
    categorized = pipeline.add_stunting_category(raw_data)
    values_2023 = categorized.loc[categorized["Tahun"].eq(2023)]

    assert set(summary["Level Wilayah"]) == {"Wilayah Indonesia", "Pulau", "Tipe Wilayah"}
    assert set(summary["Statistik"]) == {"mean", "std"}
    assert {"Kepadatan Penduduk Normal", "PDRB per Kapita Normal"} <= set(normalized.columns)
    assert normalized["Tahun"].eq(2023).all()
    assert values_2023.loc[values_2023["Prevalensi Stunting (%)"].lt(0.20), "Kategori Stunting"].eq("Rendah").all()
    assert values_2023.loc[values_2023["Prevalensi Stunting (%)"].between(0.20, 0.30, inclusive="both"), "Kategori Stunting"].eq("Sedang").all()
    assert values_2023.loc[values_2023["Prevalensi Stunting (%)"].gt(0.30), "Kategori Stunting"].eq("Tinggi / Kritis").all()


def test_task_five_outliers_and_pearson_correlation_are_requested_shapes():
    raw_data = pd.read_excel(Path(__file__).parents[1] / "Dataset_Kesehatan_Real_Names.xlsx")

    outliers = pipeline.find_iqr_outliers(raw_data)
    correlation = pipeline.health_correlation(raw_data)

    assert outliers["Tahun"].eq(2023).all()
    assert set(outliers["Variabel"]) == {"PDRB per Kapita (Juta Rp)", "Kepadatan Penduduk (Jiwa/km2)"}
    assert correlation.shape == (5, 5)
    assert correlation.index.equals(correlation.columns)
