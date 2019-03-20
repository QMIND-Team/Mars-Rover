import os
import csv

with open('train_masks.csv', 'w', newline = '') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(946):
        writer.writerow(["{}.png".format(i+1)])
