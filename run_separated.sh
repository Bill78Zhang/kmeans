rm results1.csv ;
for i in {1..100}; do python kmeans.py "separated groups of points ">> results1.csv;done
