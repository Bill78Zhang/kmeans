rm results2.csv ;
for i in {1..100}; do python kmeans.py "MNIST">> results2.csv;done
