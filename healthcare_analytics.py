import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")
plt.style.use("ggplot")

# ==========================
# LOAD DATA
# ==========================
df = pd.read_csv("dataset/healthcare_dataset.csv")

print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())
print("Duplicate Records:", df.duplicated().sum())

# ==========================
# FEATURE ENGINEERING
# ==========================
df["Date of Admission"] = pd.to_datetime(df["Date of Admission"])
df["Discharge Date"] = pd.to_datetime(df["Discharge Date"])

df["Length_of_Stay"] = (
    df["Discharge Date"] - df["Date of Admission"]
).dt.days

df["Admission_Month"] = df["Date of Admission"].dt.month_name()

df["Age_Group"] = pd.cut(
    df["Age"],
    bins=[0,18,35,50,65,100],
    labels=["0-18","19-35","36-50","51-65","65+"]
)

df["Billing_Category"] = pd.qcut(
    df["Billing Amount"],
    3,
    labels=["Low","Medium","High"]
)

# ==========================
# CHARTS
# ==========================

# 1 Age
plt.figure(figsize=(8,5))
plt.hist(df["Age"],bins=20)
plt.title("Age Distribution")
plt.show()

# 2 Gender
df["Gender"].value_counts().plot(kind="bar",title="Gender Distribution")
plt.show()

# 3 Medical Condition
df["Medical Condition"].value_counts().plot(kind="bar",figsize=(10,5),title="Medical Condition Distribution")
plt.xticks(rotation=45)
plt.show()

# 4 Blood Group
df["Blood Type"].value_counts().plot(kind="bar",title="Blood Group Distribution")
plt.show()

# 5 Admission Type
plt.figure(figsize=(6,6))
df["Admission Type"].value_counts().plot(kind="pie",autopct="%1.1f%%")
plt.ylabel("")
plt.title("Admission Type")
plt.show()

# 6 Insurance
df["Insurance Provider"].value_counts().plot(kind="bar",figsize=(10,5),title="Insurance Provider")
plt.xticks(rotation=45)
plt.show()

# 7 Top Hospitals by Patient Count
(df["Hospital"].value_counts().head(10)).plot(kind="bar",figsize=(10,5),title="Top Hospitals by Patient Count")
plt.xticks(rotation=60)
plt.show()

# 8 Top Hospitals by Revenue
(df.groupby("Hospital")["Billing Amount"].sum().sort_values(ascending=False).head(10)).plot(
    kind="bar",figsize=(10,5),title="Top Hospitals by Revenue")
plt.xticks(rotation=60)
plt.show()

# 9 Monthly Admissions
order=["January","February","March","April","May","June","July","August","September","October","November","December"]
df["Admission_Month"].value_counts().reindex(order).plot(kind="line",marker="o",title="Monthly Admissions")
plt.show()

# 10 Length of Stay
plt.figure(figsize=(8,5))
plt.hist(df["Length_of_Stay"],bins=20)
plt.title("Length of Stay")
plt.show()

# 11 Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df[["Age","Billing Amount","Room Number","Length_of_Stay"]].corr(),annot=True,cmap="coolwarm")
plt.show()

# 12 Billing Outliers
plt.figure(figsize=(6,5))
plt.boxplot(df["Billing Amount"])
plt.title("Billing Outliers")
plt.show()

# 13 LOS Outliers
plt.figure(figsize=(6,5))
plt.boxplot(df["Length_of_Stay"])
plt.title("Length of Stay Outliers")
plt.show()

# ==========================
# KMEANS
# ==========================
features=df[["Age","Billing Amount","Length_of_Stay"]]
scaled=MinMaxScaler().fit_transform(features)

kmeans=KMeans(n_clusters=3,random_state=42,n_init=10)
df["Patient_Cluster"]=kmeans.fit_predict(scaled)

plt.figure(figsize=(8,6))
plt.scatter(df["Age"],df["Billing Amount"],c=df["Patient_Cluster"])
plt.title("Patient Segmentation")
plt.xlabel("Age")
plt.ylabel("Billing Amount")
plt.show()

# ==========================
# HOSPITAL PERFORMANCE SCORE
# ==========================
hospital=df.groupby("Hospital").agg(
    Total_Patients=("Name","count"),
    Avg_Billing=("Billing Amount","mean"),
    Avg_LOS=("Length_of_Stay","mean"),
    Emergency=("Admission Type",lambda x:(x=="Emergency").sum())
).reset_index()

scaler=MinMaxScaler()
hospital[["P","B","L","E"]]=scaler.fit_transform(
    hospital[["Total_Patients","Avg_Billing","Avg_LOS","Emergency"]]
)

hospital["Hospital_Performance_Score"]=(
    0.35*hospital["P"]+
    0.25*hospital["B"]+
    0.15*hospital["L"]+
    0.25*hospital["E"]
)

print(hospital.sort_values("Hospital_Performance_Score",ascending=False).head())

# ==========================
# EXPORT
# ==========================
df.to_csv("healthcare_analytics_final.csv",index=False)
print("Project Completed Successfully")
