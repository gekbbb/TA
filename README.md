# USER GUIDE ANALISIS SENTIMEN BERBASIS WEB PADA PRODUK KECANTIKAN DENGAN PENDEKATAN DEEP LEARNING

Untuk membuat penelitian ini terdapat tiga tahap utama, yaitu akuisisi data, pembuatan analisis sentimen menggunakan LSTM, dan pembuatan website

## SCRAPING
Pengumpulan data dilakukan melalui proses scraping menggunakan library python yaitu BeautifulSoup4. Data yang digunakan pada penelitian ini adalah dataset ulasan produk kategori fragrance merek lokal yang diambil dari situs https://femaledaily.com/.

## ANALISIS SENTIMEN MENGGUNAKAN LSTM
Pembuatan analisis sentimen menggunakan LSTM melalui beberapa tahap di antaranya adalah preprocessing, konversi kalimat menggunakan lexicon dan manual labeling, embedding menggunakan IndoBERT, pemisahan data (data train & test), pembuatan dan pelatihan model menggunakan LSTM, dan pengujian akurasi sistem menggunakan confusion matrix.

## WEBSITE
Pembuatan website website menggunakan Streamlit. Website meminta input berupa link ulasan produk fragrance merek lokal dari website Female Daily, lalu ouput-nya akan berupa analisis sentimen ulasan produk fragrance tersebut (positif, netral, negatif). Fitur lain dari website ini dapat menampilkan persentase hasil masing-masing sentimen dan rating berdasarkan hasil model LSTM.

# REQUIREMENTS

Berikut adalah library dan environment yang diperlukan dalam membuat penelitian ini:

## SCRAPING

```
pip install beautifulsoup4
pip install urllib3

from bs4 import BeautifulSoup
from urllib.request import urlopen
import pandas as pd
```

## ANALISIS SENTIMEN MENGGUNAKAN LSTM

```
pip install nltk
pip install keras
pip install numpy
pip install pandas
pip install seaborn
pip install tensorflow
pip install matplotlib
pip install clean-text
pip install deep_translator
pip install beautifulsoup4
pip install transformers==3.5.1
pip install sklearn

import time
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import keras
import string
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline

from cleantext import clean
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from deep_translator import GoogleTranslator
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import to_categorical
from tensorflow.python.keras.utils import np_utils
from transformers import BertTokenizer, BertModel
from sklearn.metrics import confusion_matrix, classification_report
```

## WEBSITE

```
pip install regex
pip install nltk
pip install keras
pip install numpy
pip install pandas
pip install tensorflow
pip install clean-text
pip install beautifulsoup4
pip install transformers==3.5.1
pip install sklearn

import re
import time
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import keras
import string
import numpy as np
import pandas as pd
import tensorflow as tf

from textblob import TextBlob
import streamlit as st
from ml_utils import inference
from urllib.request import urlopen
from cleantext import clean
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from keras.utils import to_categorical
from transformers import BertTokenizer, BertModel
```
