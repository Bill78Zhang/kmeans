rm results.csv ;
for i in {1..100}; do python kmeans.py "iris.data.txt">> results.csv;done
