# Overview

This is a pretty generic scraper for taking images off of a Google image search. Main usage for this project is collecting
images of tennis balls and generic backgrounds, used in constructing our training data for our model.

# Scraper Usage

This is the generic CLI call made to run this program:

`python scraper.py [-d --directory saveDirectory] [-s --search searchItem] [-n --num_images numImagesToScrape]`

A quick overview of what each of those optional flags is used for:
* `-d` or `--directory`: Specify directory scraped images will be saved in [i.e. "~/Documents/"]. The default argument
will just be the directory the program is ran (probably a good idea to set this parameter manually).

* `-s` or `--search`: Image you're scraping for [i.e. "tennis ball"]. The default argument for this will be "tennis ball",
given the context of the project.
* `-n` or `--num_images`: Number of images to scrape [i.e. 5]: Default argument is 1.

# Future Additions

* This program will soon use randomly selected proxies during queries.

# Reference

This code was inspired by https://gist.github.com/genekogan/ebd77196e4bf0705db51f86431099e57, and the comments associated with
this repository.
