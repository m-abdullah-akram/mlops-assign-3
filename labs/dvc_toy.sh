#!/usr/bin/env bash
# Task 1.4 TOY PART - run line by line in a NEW EMPTY folder and screenshot each step.
set -e
mkdir dvc-toy && cd dvc-toy
git init && dvc init
printf "id,value\n1,10\n2,20\n" > data.csv
dvc add data.csv                               # creates data.csv.dvc and git-ignores data.csv
git add data.csv.dvc .gitignore && git commit -m "v1: initial CSV"
printf "3,30\n" >> data.csv
dvc add data.csv && git add data.csv.dvc && git commit -m "v2: added a row"
git log --oneline
git checkout HEAD~1 -- data.csv.dvc && dvc checkout   # back to the OLD version
cat data.csv
git checkout HEAD -- data.csv.dvc && dvc checkout      # forward again
# Remote storage: DagsHub storage (use YOUR user/repo) or a local folder
dvc remote add -d storage ../dvc-toy-remote            # local-folder remote
# DagsHub alternative:
#   dvc remote add -d origin s3://dvc            (see DagsHub "Remote > Data" tab for the exact commands)
dvc push
rm -rf data.csv .dvc/cache                     # delete local data
dvc pull && cat data.csv                       # get it back
