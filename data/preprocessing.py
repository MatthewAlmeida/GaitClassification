import pandas as pd
import numpy as np

import sklearn.preprocessing as skpp

if __name__ == "__main__":
    raw_data = pd.read_excel("DATANOTAVG01202017.xlsx",sheet_name="D_BINARIZED")

    print(raw_data.columns)

    # -----
    # Analysis performed in the excel sheet indicates
    # that the two length columns are nicely normally 
    # distributed. Gait Speed and Cadence appear normal
    # as well.
    #
    # The time columns (and asymmetry) appear to be roughly log-normally
    # distributed, so we transform them with the natural 
    # logarithm before z-score normalization.
    # -----

    X_COLS = [u'Over_85',u'Edu_No_HS', u'Edu_HS', u'Edu_Col',
       u'At_Gender',u'Race_White', u'Race_Black', u'Race_Other', u'Age_At_Visit',
       u'Gait_Spd', 
       u'Cadence', u'Step_LenL', u'Step_LenR', u'Step_TimeL', u'Step_TimeR',  
       u'Stride_TimeL', u'Stride_TimeR', u'Stance_TimeL', u'Stance_TimeR', 
       u'Swing_TimeL', u'Swing_TimeR', u'DblSupp_TimeL', u'DblSupp_TimeR', u'Asymmetry',
       ]

    target = [u'Walk_Type_ID']

    PAIN_COLS = [u'BPI_Sev', u'BPI_Interf', u'BPI_Sev_Tert',
       u'BPI_Interf_tert', u'PainWHAS3', u'PainWHAS2', u'PainWHAS1',
       u'PainWHAS0']
    
    X = raw_data[X_COLS].values
    y = raw_data[target].values

    y = np.reshape(y, (y.shape[0],))

    # Consolidate classes 1 and 2.

    y[y != 0] =1

    # Take the logarithm of the log-normally distributed features.
    # Scaling will be performed during the cross-validation process
    # so that we don't scale based on examples in the test set.

    X[:,13:] = np.log(X[:,13:])

    # There are a few 0s in the log-normally distributed columns.
    # we replace the negative infinite values with a value in range.

    X[np.isneginf(X)] = -2.0

    np.save("X.npy", X)
    np.save("y.npy", y)
