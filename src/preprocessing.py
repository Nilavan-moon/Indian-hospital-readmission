"""Load the five relational tables, merge them and prepare model-ready matrices."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

SEED = 42
TARGET = "readmitted_30d"
LEAK_COLS = ["readmitted_7d"]  # 7-day outcome is a sub-event of the target


def load_tables(data_dir):
    # keep_default_na=False: the string "None" in insurance_type means uninsured, not missing
    t = {n: pd.read_csv(f"{data_dir}/{n}.csv", keep_default_na=False, na_values=[""])
         for n in ["admissions", "patients", "diagnoses", "billing", "hospitals"]}
    return t


def quality_report(t):
    rep = {}
    for n, d in t.items():
        rep[n] = dict(rows=len(d), cols=d.shape[1], missing=int(d.isna().sum().sum()),
                      duplicates=int(d.duplicated().sum()))
    return rep


def build_dataset(t, drop_expired=True):
    a, p, d, b, h = (t[k].copy() for k in ["admissions", "patients", "diagnoses", "billing", "hospitals"])
    a = a.drop_duplicates("admission_id")
    prim = d[d.diag_rank == 1].drop_duplicates("admission_id")[["admission_id", "diag_category"]]
    ndx = d.groupby("admission_id").size().rename("n_diagnoses").reset_index()
    h = h[["hospital_id", "tier", "beds", "teaching"]].rename(columns={"tier": "hospital_tier"})
    p = p.rename(columns={"state": "patient_state"})
    df = (a.merge(p, on="patient_id", how="left").merge(h, on="hospital_id", how="left")
           .merge(b.drop(columns="bill_id"), on="admission_id", how="left")
           .merge(prim, on="admission_id", how="left").merge(ndx, on="admission_id", how="left"))
    n0 = len(df)
    if drop_expired:  # patients who died cannot be readmitted
        df = df[df.discharge_type != "Expired"]
    df["admit_date"] = pd.to_datetime(df["admit_date"])
    df["admit_year"] = df.admit_date.dt.year
    df["admit_month"] = df.admit_date.dt.month
    df["cost_per_day"] = df.total_cost_inr / df.los_days.clip(lower=1)
    df["oop_share"] = df.out_of_pocket_inr / df.total_cost_inr.clip(lower=1)
    df["teaching"] = df["teaching"].astype(int)
    df["bpl_card"] = df["bpl_card"].astype(int)
    df["diag_category"] = df["diag_category"].fillna("Unknown")
    df["n_diagnoses"] = df["n_diagnoses"].fillna(1)
    df.attrs["dropped_expired"] = n0 - len(df)
    return df.reset_index(drop=True)


ID_COLS = ["admission_id", "patient_id", "hospital_id", "admit_date", "discharge_date"]
CAT_COLS = ["admit_type", "ward_type", "discharge_type", "gender", "patient_state", "insurance_type",
            "hospital_tier", "cost_category", "diag_category"]


def feature_frame(df):
    """One-hot encode categoricals; median-impute numerics. Returns X (DataFrame), y."""
    X = df.drop(columns=ID_COLS + [TARGET] + LEAK_COLS)
    num = [c for c in X.columns if c not in CAT_COLS]
    X[num] = X[num].fillna(X[num].median())
    X[CAT_COLS] = X[CAT_COLS].fillna("Unknown")
    X = pd.get_dummies(X, columns=CAT_COLS, dtype=float)
    return X, df[TARGET].values, num


def split_by_patient(df, seed=SEED, test=0.15, val=0.15):
    """Patient-grouped stratification-free split so one patient never spans two sets."""
    g = df.patient_id.values
    gss = GroupShuffleSplit(n_splits=1, test_size=test, random_state=seed)
    tv, te = next(gss.split(df, groups=g))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val / (1 - test), random_state=seed)
    tr, va = next(gss2.split(df.iloc[tv], groups=g[tv]))
    return tv[tr], tv[va], te


def scale(X_tr, X_va, X_te, num_cols):
    sc = StandardScaler().fit(X_tr[num_cols])
    out = []
    for X in (X_tr, X_va, X_te):
        Z = X.copy()
        Z[num_cols] = sc.transform(X[num_cols])
        out.append(Z)
    return out
