#!/bin/bash

for i in 0 1 2 3 4 5 6 7 8 9; do
  python ex8${i}.py > ex8${i}_output.txt 2>&1
done