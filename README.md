# Biohack2018


# Feature selection


**Process Conservation**

Process conservation with profile
```
    python conservation.py conservation.bw BAND.bed OUTPUT.csv
```


**Process BED**

Process bedfile intersection with profile
```
    python intersect.py SORTED_FILE_STAIN SORTED_FILE_REGIONS BINS OUTPUT_CSV
```

**Visualize**
Visualize single file 
```
    python visualize.py plot foo.png file.csv
```

Visualize logfc 
```
    python visualize.py logfc foo.png file1.csv file2.csv
```

Visualize heatmap
```
    python visualize.py heatmap foo.png file1.csv file2.csv file3.csv ...
```


Neural networks
===============
Prepare data and launch NN.
```
    python prepare_ml.py
    python nn.py
```
