from pathlib import Path

import pandas as pd

import wrangling_pipeline as pipeline
from wrangling_pipeline import (
    buat_data_long,
    buat_panel_stunting,
    cari_outlier_iqr,
    filter_stunting_dan_kemiskinan_2023,
    gabungkan_dimensi_dan_fakta,
    korelasi_kesehatan,
    normalisasi_minmax_per_pulau,
    pisahkan_tabel,
    ringkas_kesehatan_per_wilayah,
    tambah_kategori_stunting,
)


def test_split_tables_creates_distinct_dimension_and_fact_tables():
    raw_data = pd.read_excel(Path(__file__).parents[1] / "Dataset_Kesehatan_Real_Names.xlsx")
    expected_facts = raw_data.drop(
        columns=["Provinsi", "Pulau", "Wilayah Indonesia", "Tipe Wilayah"]
    )

    dimension, facts = pisahkan_tabel(raw_data)

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

    long_data = buat_data_long(raw_data)

    assert list(long_data.columns) == ["Kabupaten/Kota", "Tahun", "Nama Indikator", "Nilai"]
    assert long_data.shape == (59110, 4)
    assert long_data["Nama Indikator"].nunique() == 23


def test_stunting_panel_has_requested_year_columns():
    raw_data = pd.read_excel(Path(__file__).parents[1] / "Dataset_Kesehatan_Real_Names.xlsx")

    panel = buat_panel_stunting(raw_data)

    assert list(panel.columns) == [2015, 2018, 2019, 2021, 2023]
    assert panel.shape == (514, 5)


def test_2023_filter_and_left_join_preserve_expected_rows():
    raw_data = pd.read_excel(Path(__file__).parents[1] / "Dataset_Kesehatan_Real_Names.xlsx")
    dimension, facts = pisahkan_tabel(raw_data)

    filtered = filter_stunting_dan_kemiskinan_2023(raw_data)
    merged = gabungkan_dimensi_dan_fakta(dimension, facts)

    assert filtered["Tahun"].eq(2023).all()
    assert filtered["Prevalensi Stunting (%)"].gt(0.30).all()
    assert filtered["Penduduk Miskin (%)"].gt(0.10).all()
    assert merged.shape == (2570, 29)


def test_task_four_summaries_normalization_and_categories():
    raw_data = pd.read_excel(Path(__file__).parents[1] / "Dataset_Kesehatan_Real_Names.xlsx")

    summary = ringkas_kesehatan_per_wilayah(raw_data)
    normalized = normalisasi_minmax_per_pulau(raw_data)
    categorized = tambah_kategori_stunting(raw_data)
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

    outliers = cari_outlier_iqr(raw_data)
    correlation = korelasi_kesehatan(raw_data)

    assert outliers["Tahun"].eq(2023).all()
    assert set(outliers["Variabel"]) == {"PDRB per Kapita (Juta Rp)", "Kepadatan Penduduk (Jiwa/km2)"}
    assert correlation.shape == (5, 5)
    assert correlation.index.equals(correlation.columns)
