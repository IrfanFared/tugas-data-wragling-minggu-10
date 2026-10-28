"""Pipeline transformasi dan pengolahan data kesehatan Indonesia.

"""

import pandas as pd

# Definisi nama kolom dimensi wilayah sesuai ketentuan tugas
KOLOM_DIMENSI = [
    "Kabupaten/Kota",
    "Provinsi",
    "Pulau",
    "Wilayah Indonesia",
    "Tipe Wilayah",
]

KOLOM_LOKASI = KOLOM_DIMENSI[1:]
KOLOM_ID = ["Kabupaten/Kota", "Tahun"]
KOLOM_STUNTING = "Prevalensi Stunting (%)"
KOLOM_KEMISKINAN = "Penduduk Miskin (%)"
KOLOM_AKI = "AKI (per 100rb Kelahiran)"
KOLOM_KEPADATAN = "Kepadatan Penduduk (Jiwa/km2)"
KOLOM_PDRB = "PDRB per Kapita (Juta Rp)"

# Lima indikator untuk analisis korelasi (Task 5.2):
# - 2 indikator sosio-ekonomi : Penduduk Miskin (%), PDRB per Kapita (Juta Rp)
# - 3 indikator pelayanan & hasil kesehatan : Prevalensi Stunting (%), Cakupan BPJS/JKN (%), Rasio Bed RS (per 1000 Jiwa)
KOLOM_KORELASI = [
    KOLOM_KEMISKINAN,
    KOLOM_PDRB,
    KOLOM_STUNTING,
    "Cakupan BPJS/JKN (%)",
    "Rasio Bed RS (per 1000 Jiwa)",
]


def pisahkan_tabel(data):
    
    dimensi = data.loc[:, KOLOM_DIMENSI].drop_duplicates().reset_index(drop=True)
    fakta = data.drop(columns=KOLOM_DIMENSI[1:])
    return dimensi, fakta


def buat_data_long(data):
    indikator = [
        kolom for kolom in data.columns if kolom not in KOLOM_ID + KOLOM_LOKASI
    ]
    return data.melt(
        id_vars=KOLOM_ID,
        value_vars=indikator,
        var_name="Nama Indikator",
        value_name="Nilai",
    )


def buat_panel_stunting(data):
    daftar_tahun = [2015, 2018, 2019, 2021, 2023]
    panel = data.pivot(
        index="Kabupaten/Kota", columns="Tahun", values=KOLOM_STUNTING
    )
    return panel.reindex(columns=daftar_tahun)


def filter_stunting_dan_kemiskinan_2023(data):
    return data.loc[
        data["Tahun"].eq(2023)
        & data[KOLOM_STUNTING].gt(0.30)
        & data[KOLOM_KEMISKINAN].gt(0.10)
    ].copy()


def gabungkan_dimensi_dan_fakta(dimensi, fakta):
  
    return dimensi.merge(fakta, on="Kabupaten/Kota", how="left", validate="one_to_many")


def ringkas_kesehatan_per_wilayah(data):
    ringkasan_list = []
    for kolom_wilayah in ["Wilayah Indonesia", "Pulau", "Tipe Wilayah"]:
        grup = data.groupby(kolom_wilayah)[[KOLOM_AKI, KOLOM_STUNTING]].agg(["mean", "std"])
        grup.columns = [f"{metrik} {statistik}" for metrik, statistik in grup.columns]
        ringkasan = grup.reset_index().melt(
            id_vars=kolom_wilayah, var_name="Ukuran", value_name="Nilai"
        )
        metrik_dan_stat = ringkasan["Ukuran"].str.rsplit(" ", n=1, expand=True)
        metrik = metrik_dan_stat[0]
        stat = metrik_dan_stat[1]
        ringkasan = ringkasan.assign(
            **{
                "Level Wilayah": kolom_wilayah,
                "Wilayah": ringkasan[kolom_wilayah],
                "Indikator": metrik,
                "Statistik": stat,
            }
        )
        ringkasan_list.append(ringkasan[["Level Wilayah", "Wilayah", "Indikator", "Statistik", "Nilai"]])
    return pd.concat(ringkasan_list, ignore_index=True)


def normalisasi_minmax_per_pulau(data):
    
    ternormalisasi = data.loc[data["Tahun"].eq(2023)].copy()
    for sumber, target in [
        (KOLOM_KEPADATAN, "Kepadatan Penduduk Normal"),
        (KOLOM_PDRB, "PDRB per Kapita Normal"),
    ]:
        nilai_min = ternormalisasi.groupby("Pulau")[sumber].transform("min")
        rentang = ternormalisasi.groupby("Pulau")[sumber].transform("max") - nilai_min
        ternormalisasi[target] = (ternormalisasi[sumber] - nilai_min).div(rentang).fillna(0.0)
    return ternormalisasi


def tambah_kategori_stunting(data):

    hasil = data.copy()
    nilai = hasil[KOLOM_STUNTING]
    hasil["Kategori Stunting"] = pd.NA
    mask_2023 = hasil["Tahun"].eq(2023)
    hasil.loc[mask_2023 & nilai.lt(0.20), "Kategori Stunting"] = "Rendah"
    hasil.loc[
        mask_2023 & nilai.between(0.20, 0.30, inclusive="both"), "Kategori Stunting"
    ] = "Sedang"
    hasil.loc[mask_2023 & nilai.gt(0.30), "Kategori Stunting"] = "Tinggi / Kritis"
    return hasil


def cari_outlier_iqr(data):
    
    sub = data.loc[data["Tahun"].eq(2023)].copy()
    daftar_outlier = []
    for kolom in [KOLOM_PDRB, KOLOM_KEPADATAN]:
        q1, q3 = sub[kolom].quantile([0.25, 0.75])
        iqr = q3 - q1
        mask = sub[kolom].lt(q1 - 1.5 * iqr) | sub[kolom].gt(q3 + 1.5 * iqr)
        daftar_outlier.append(
            sub.loc[mask, ["Kabupaten/Kota", "Tahun", kolom]]
            .rename(columns={kolom: "Nilai"})
            .assign(Variabel=kolom)
        )
    return pd.concat(daftar_outlier, ignore_index=True)


def korelasi_kesehatan(data):

    return data.loc[:, KOLOM_KORELASI].corr(method="pearson")
