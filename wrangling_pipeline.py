"""Reusable transformations for the health dataset."""

import pandas as pd

DIMENSION_COLUMNS = [
    "Kabupaten/Kota",
    "Provinsi",
    "Pulau",
    "Wilayah Indonesia",
    "Tipe Wilayah",
]

LOCATION_COLUMNS = DIMENSION_COLUMNS[1:]
ID_COLUMNS = ["Kabupaten/Kota", "Tahun"]
STUNTING_COLUMN = "Prevalensi Stunting (%)"
POVERTY_COLUMN = "Penduduk Miskin (%)"
AKI_COLUMN = "AKI (per 100rb Kelahiran)"
DENSITY_COLUMN = "Kepadatan Penduduk (Jiwa/km2)"
PDRB_COLUMN = "PDRB per Kapita (Juta Rp)"
CORRELATION_COLUMNS = [
    AKI_COLUMN,
    "AKB (per 1000 Kelahiran)",
    STUNTING_COLUMN,
    POVERTY_COLUMN,
    PDRB_COLUMN,
]


def split_tables(data):
    """Return distinct regional dimensions and health fact records."""
    dimension = data.loc[:, DIMENSION_COLUMNS].drop_duplicates().reset_index(drop=True)
    facts = data.drop(columns=DIMENSION_COLUMNS[1:])
    return dimension, facts


def make_long_data(data):
    """Melt every health indicator into the required four-column tidy format."""
    indicators = [
        column for column in data.columns if column not in ID_COLUMNS + LOCATION_COLUMNS
    ]
    return data.melt(
        id_vars=ID_COLUMNS,
        value_vars=indicators,
        var_name="Nama Indikator",
        value_name="Nilai",
    )


def make_stunting_panel(data):
    """Return stunting prevalence by district/city for the five observed years."""
    years = [2015, 2018, 2019, 2021, 2023]
    panel = data.pivot(
        index="Kabupaten/Kota", columns="Tahun", values=STUNTING_COLUMN
    )
    return panel.reindex(columns=years)


def filter_stunting_and_poverty_2023(data):
    """Select 2023 regions whose stunting and poverty exceed the thresholds."""
    return data.loc[
        data["Tahun"].eq(2023)
        & data[STUNTING_COLUMN].gt(0.30)
        & data[POVERTY_COLUMN].gt(0.10)
    ].copy()


def merge_dimension_and_facts(dimension, facts):
    """Restore regional fields with a cardinality-safe left join."""
    return dimension.merge(facts, on="Kabupaten/Kota", how="left", validate="one_to_many")


def summarize_health_by_region(data):
    """Calculate mean and standard deviation of AKI and stunting at three levels."""
    summaries = []
    for region_column in ["Wilayah Indonesia", "Pulau", "Tipe Wilayah"]:
        grouped = data.groupby(region_column)[[AKI_COLUMN, STUNTING_COLUMN]].agg(["mean", "std"])
        grouped.columns = [f"{metric} {statistic}" for metric, statistic in grouped.columns]
        summary = grouped.reset_index().melt(
            id_vars=region_column, var_name="Ukuran", value_name="Nilai"
        )
        metric_statistic = summary["Ukuran"].str.rsplit(" ", n=1, expand=True)
        metric = metric_statistic[0]
        statistic = metric_statistic[1]
        summary = summary.assign(
            **{
                "Level Wilayah": region_column,
                "Wilayah": summary[region_column],
                "Indikator": metric,
                "Statistik": statistic,
            }
        )
        summaries.append(summary[["Level Wilayah", "Wilayah", "Indikator", "Statistik", "Nilai"]])
    return pd.concat(summaries, ignore_index=True)


def minmax_by_island(data):
    """Min-max normalize density and PDRB inside each island for 2023."""
    normalized = data.loc[data["Tahun"].eq(2023)].copy()
    for source, target in [
        (DENSITY_COLUMN, "Kepadatan Penduduk Normal"),
        (PDRB_COLUMN, "PDRB per Kapita Normal"),
    ]:
        minimum = normalized.groupby("Pulau")[source].transform("min")
        span = normalized.groupby("Pulau")[source].transform("max") - minimum
        normalized[target] = (normalized[source] - minimum).div(span).fillna(0.0)
    return normalized


def add_stunting_category(data):
    """Classify 2023 stunting as rendah, sedang, or tinggi/kritis."""
    categorized = data.copy()
    values = categorized[STUNTING_COLUMN]
    categorized["Kategori Stunting"] = pd.NA
    mask_2023 = categorized["Tahun"].eq(2023)
    categorized.loc[mask_2023 & values.lt(0.20), "Kategori Stunting"] = "Rendah"
    categorized.loc[
        mask_2023 & values.between(0.20, 0.30, inclusive="both"), "Kategori Stunting"
    ] = "Sedang"
    categorized.loc[mask_2023 & values.gt(0.30), "Kategori Stunting"] = "Tinggi / Kritis"
    return categorized


def find_iqr_outliers(data):
    """Return 2023 density and PDRB observations outside their 1.5-IQR fences."""
    subset = data.loc[data["Tahun"].eq(2023)].copy()
    outliers = []
    for column in [PDRB_COLUMN, DENSITY_COLUMN]:
        q1, q3 = subset[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        mask = subset[column].lt(q1 - 1.5 * iqr) | subset[column].gt(q3 + 1.5 * iqr)
        outliers.append(
            subset.loc[mask, ["Kabupaten/Kota", "Tahun", column]]
            .rename(columns={column: "Nilai"})
            .assign(Variabel=column)
        )
    return pd.concat(outliers, ignore_index=True)


def health_correlation(data):
    """Return the requested five-variable Pearson correlation matrix."""
    return data.loc[:, CORRELATION_COLUMNS].corr(method="pearson")
