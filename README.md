## SCRAPING

```
pip install beautifulsoup4
pip install urllib3

from bs4 import BeautifulSoup
from urllib.request import urlopen
import pandas as pd
```

## LSTM ANALISIS SENTIMEN

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
