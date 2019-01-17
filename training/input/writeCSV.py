import os
import csv

files = os.listdir(os.getcwd() + '\\train')
newFiles = []
 for file in files:
    newFiles.append([file])

with open('train_masks.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    for file in newFiles:
        writer.writerow(file)
