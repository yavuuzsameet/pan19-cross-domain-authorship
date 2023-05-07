# pan19-cross-domain-authorship

### Import nltk
```
pip install nltk
```

### Preprocessing
```
python3 preprocess.py -i pan19-cross-domain-authorship-attribution-training-dataset-2019-01-23 -o preprocessed
```

### Import sklearn
```
pip install scikit-learn
```

### Model
```
python3 model.py -i preprocessed/problem00001
```