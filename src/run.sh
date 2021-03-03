# /bin/sh bash
# implement from /src

C:/Users/College/.conda/envs/myml/python.exe "../src/clean_data.py" --ifile RAW_DATA --ofile CLEAN_DATA
C:/Users/College/.conda/envs/myml/python.exe "../src/feature.py" --ifile CLEAN_DATA --ofile PROCESSED_DATA
C:/Users/College/.conda/envs/myml/python.exe "../src/create_folds.py" --ifile PROCESSED_DATA --ofile FOLDED_DATA

