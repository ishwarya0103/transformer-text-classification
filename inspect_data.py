"""
Data inspection script for SPR 2026 Mammography BI-RADS Classification.
Copy cells into a playground notebook as needed.
"""

# %% --- Load data ---
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
submission = pd.read_csv("submission.csv")

print(f"Train shape: {train.shape}")
print(f"Test shape:  {test.shape}")
print(f"Submission shape: {submission.shape}")

# %% --- Schema & types ---
print("\n=== Train info ===")
print(train.dtypes)
print(f"\nNull counts:\n{train.isnull().sum()}")

print("\n=== Test info ===")
print(test.dtypes)
print(f"\nNull counts:\n{test.isnull().sum()}")

# %% --- Target distribution ---
print("\n=== Target (BI-RADS) distribution ===")
target_counts = train["target"].value_counts().sort_index()
print(target_counts)
print(f"\nTarget proportions:\n{target_counts / len(train)}")

fig, ax = plt.subplots(figsize=(8, 4))
target_counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
ax.set_xlabel("BI-RADS Category")
ax.set_ylabel("Count")
ax.set_title("Target Distribution (Train)")
for i, (cat, count) in enumerate(target_counts.items()):
    ax.text(i, count + len(train) * 0.005, f"{count}\n({count/len(train)*100:.1f}%)",
            ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.show()

# %% --- Report text statistics ---
train["report_len"] = train["report"].str.len()
train["report_words"] = train["report"].str.split().apply(len)

test["report_len"] = test["report"].str.len()
test["report_words"] = test["report"].str.split().apply(len)

print("\n=== Report length (chars) ===")
print(train["report_len"].describe())

print("\n=== Report length (words) ===")
print(train["report_words"].describe())

# %% --- Report length distribution by target ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

train.boxplot(column="report_words", by="target", ax=axes[0])
axes[0].set_title("Word count by BI-RADS category")
axes[0].set_xlabel("BI-RADS Category")
axes[0].set_ylabel("Word count")

train["report_len"].hist(bins=50, ax=axes[1], color="steelblue", edgecolor="black")
axes[1].set_title("Report length distribution (chars)")
axes[1].set_xlabel("Character count")
axes[1].set_ylabel("Frequency")

plt.suptitle("")
plt.tight_layout()
plt.show()

# %% --- Sample reports per category ---
for cat in sorted(train["target"].unique()):
    print(f"\n{'='*60}")
    print(f"BI-RADS {cat} — Sample report:")
    print("=" * 60)
    sample = train[train["target"] == cat].iloc[0]
    print(sample["report"][:500])
    print("...")

# %% --- Language detection (quick check) ---
print("\n=== Language sample (first 5 reports) ===")
for _, row in train.head(5).iterrows():
    snippet = row["report"][:80].replace("\n", " ")
    print(f"  [{row['ID']}] {snippet}...")

# %% --- Check for duplicate reports ---
n_dupes = train["report"].duplicated().sum()
print(f"\nDuplicate reports in train: {n_dupes} ({n_dupes/len(train)*100:.2f}%)")

# %% --- Check test/train overlap ---
overlap = set(test["ID"]) & set(train["ID"])
print(f"ID overlap between train and test: {len(overlap)}")

report_overlap = set(test["report"]) & set(train["report"])
print(f"Report text overlap between train and test: {len(report_overlap)}")

# %% --- Submission sanity check ---
print(f"\nSubmission IDs match test IDs: {set(submission['ID']) == set(test['ID'])}")
print(f"Submission target placeholder values: {submission['target'].unique()}")

# %% --- Cleanup temp columns ---
train.drop(columns=["report_len", "report_words"], inplace=True)
test.drop(columns=["report_len", "report_words"], inplace=True)

print("\nDone! Data inspection complete.")
