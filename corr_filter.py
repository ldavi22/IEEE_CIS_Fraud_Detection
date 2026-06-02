import numpy as np

def filter(x_train, y_train, threshold):
    corr_matrix = np.abs(x_train.corr())
    cols = corr_matrix.columns.tolist()

    to_drop = set()

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if cols[i] in to_drop or cols[j] in to_drop:
                continue
            if corr_matrix.loc[cols[i], cols[j]] > threshold:
                corr_i = abs(x_train[cols[i]].corr(y_train))
                corr_j = abs(x_train[cols[j]].corr(y_train))
                if corr_i >= corr_j:
                    to_drop.add(cols[j])
                else:
                    to_drop.add(cols[i])

    return x_train.drop(columns=list(to_drop))