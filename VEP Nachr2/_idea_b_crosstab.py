import pandas as pd

df = pd.read_excel("data/raw/final_mapped.xlsx", sheet_name="unique_variants")

# Keep only the 3 clean direction labels (mirror loader's drop_ambiguous)
EFFECTS = ["LOF", "No net effect", "GOF"]
df = df[df["Effect"].isin(EFFECTS)].copy()

print(f"variants with clean direction label: {len(df)}")
print(f"distinct Pathology values: {df['Pathology'].nunique()}")
print(f"missing Pathology: {df['Pathology'].isna().sum()}")

# Standardize a bit: strip whitespace, normalize a few known synonyms
df["Pathology"] = df["Pathology"].astype(str).str.strip()

ct = pd.crosstab(df["Pathology"], df["Effect"])
ct["total"] = ct.sum(axis=1)
ct = ct.sort_values("total", ascending=False)

# For each pathology, is it "clean" (>=2/3 of variants share one direction) or "mixed"?
def dominant(row):
    vals = {"LOF": row.get("LOF", 0), "GOF": row.get("GOF", 0),
            "No net effect": row.get("No net effect", 0)}
    d = max(vals, key=vals.get)
    frac = vals[d] / row["total"]
    return d, frac

print("\n=== Pathology x Effect (sorted by # variants) ===")
print(ct.to_string())

print("\n=== per-pathology direction purity ===")
rows = []
for patho, row in ct.iterrows():
    if patho == "nan" or row["total"] < 1:
        continue
    d, frac = dominant(row)
    rows.append({"pathology": patho, "n": int(row["total"]),
                 "dominant": d, "purity": round(frac, 2)})
purity = pd.DataFrame(rows).sort_values("n", ascending=False)
print(purity.to_string(index=False))

# Coverage: how many variants live in a "clean" pathology (>=80% one direction)?
clean = purity[(purity["purity"] >= 0.8)]
clean_n = int(clean["n"].sum())
print(f"\nvariants in pathologies with >=80% single-direction purity: {clean_n}/{len(df)} "
      f"({clean_n/len(df):.1%})")
