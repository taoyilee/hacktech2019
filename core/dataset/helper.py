from sklearn.preprocessing import MinMaxScaler


def normalize_signal(x):
    scaler = MinMaxScaler()
    return scaler.fit_transform(x)
