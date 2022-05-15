# -*- coding: utf-8 -*-
"""KaggleTrial.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rDtc4wnWow7QMN2aoExKVxQrSMasqsc4
"""

#install kaggle
!pip install -q kaggle

from google.colab import files
files.upload()

! cp kaggle.json ~/.kaggle/

! chmod 600 ~/.kaggle/kaggle.json

! kaggle datasets list

! kaggle competitions download -c tabular-playground-series-may-2022

! kaggle competitions download -c tabular-playground-series-may-2022

! unzip tabular-playground-series-may-2022.zip

import pandas as pd
import numpy as np

testDf = pd.read_csv('test.csv')
idDf = testDf[['id']];

idDf['target'] = np.random.rand(700000,1)

idDf.to_csv('submission.csv');
print(idDf);

