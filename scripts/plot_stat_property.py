import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if __name__ == "__main__":
    df = pd.read_excel("plot/mitdb_summary.xlsx")
    print(df.keys())
    print(len(df))
    print(df.loc[df["Max_RRi"].isna()])
    sns.distplot(df["Max_RRi"], label="Max_RRi", norm_hist=True)
    sns.distplot(df["Min_RRi"], label="Min_RRi", norm_hist=True)
    plt.grid()
    plt.legend()
    plt.title("RR interval Histogram")
    plt.xlabel("RR interval (s)")
    plt.xlim([0, 3])
    plt.savefig("plot/rri_hist.png")
    plt.close()

    sns.distplot(df["Max_HRV_perc"], bins=120)
    plt.grid()
    plt.xlim([0, 40])
    plt.savefig("plot/max_hrv_hist.png")
    plt.close()

    sns.distplot(df.loc[(df["Max_RRi"] < 1.2) & (df["Min_RRi"] > 0.6), "Max_HRV_perc"], norm_hist=True, bins=120)
    plt.grid()
    plt.title("HRV percentage Histogram")
    plt.xlabel("HRV percentage (%)")
    plt.xlim([0, 6])
    plt.savefig("plot/normal_only_hrv_hist.png")
    plt.close()

    df.loc[(df["Max_RRi"] < 1.2) & (df["Min_RRi"] > 0.6) & (df["Max_HRV_perc"] < 5), "Arrhythmia"] = False
    df.loc[(df["Max_RRi"] < 1.2) & (df["Min_RRi"] > 0.6) & (df["Max_HRV_perc"] >= 5), "Arrhythmia"] = True
    df.loc[(df["Max_RRi"] >= 1.2) | (df["Min_RRi"] <= 0.6), "Arrhythmia"] = True
    df.to_excel("plot/mitdb_labeled.xlsx")

    # df2 = df.loc[(df["Max_RRi"] < 1.2) & (df["Min_RRi"] > 0.6) & (df["Max_HRV_perc"] > 3)]
    # print(len(df2))
    # df2.to_excel("plot/abnormal_in_normal.xlsx")
    #
    # df3 = df.loc[(df["Max_RRi"] < 1.2) & (df["Min_RRi"] > 0.6) & (df["Max_HRV_perc"] < 1)]
    # print(len(df3), f"{100 * len(df3) / len(df):.2f}%")
    #
    # df3 = df.loc[(df["Max_RRi"] < 1.2) & (df["Min_RRi"] > 0.6) & (df["Max_HRV_perc"] >= 1) & (df["Max_HRV_perc"] <= 5)]
    # print(len(df3))
    # df3.to_excel("plot/pending_review.xlsx")
