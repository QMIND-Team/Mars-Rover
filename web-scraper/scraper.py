"""
Web scraper for QMIND's Mars Rover Team
Adapted from https://gist.github.com/genekogan/ebd77196e4bf0705db51f86431099e57
author: Tyler Mainguy
TODO: Add proxy to Request object
TODO: Automate proxy choice. Potentially another program to scrape values off of free-proxy-list, or a similar domain.
"""

import argparse
import json
import itertools
import logging
import os
import uuid
from urllib.request import urlopen, Request

from bs4 import BeautifulSoup


def configure_logging():
    logger = logging.getLogger(name="./logger.log")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter('[%(asctime)s %(levelname)s %(module)s]: %(message)s'))
    logger.addHandler(handler)
    return logger


logger = configure_logging()

REQUEST_HEADER = {
    'User-Agent': "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  + "Chrome/43.0.2357.134 Safari/537.36"}


#Get soup given url and header
def get_soup(url, header):
    request_data = Request(url, headers=header)
    # request_data.set_proxy("170.84.60.222:42981", "http")
    response = urlopen(request_data)
    return BeautifulSoup(response, 'html.parser')


#Search query for image to be searched on google
def get_query_url(query):
    return "https://www.google.co.in/search?q=%s&source=lnms&tbm=isch" % query


#Pulls images from soup
def extract_images_from_soup(soup):
    image_elements = soup.find_all("div", {"class": "rg_meta"})
    metadata_dicts = (json.loads(e.text) for e in image_elements)
    link_type_records = ((d["ou"], d["ity"]) for d in metadata_dicts)
    return link_type_records


#Extracts the images from the soup request
def extract_images(query, num_images):
    url = get_query_url(query)
    logger.info("Souping")
    soup = get_soup(url, REQUEST_HEADER)
    logger.info("Extracting image urls")
    link_type_records = extract_images_from_soup(soup)
    return itertools.islice(link_type_records, num_images)


#Raw request for image
def get_raw_image(url):
    req = Request(url, headers=REQUEST_HEADER)
    resp = urlopen(req)
    return resp.read()


#Save image to filesys.
def save_image(raw_image, image_type, save_directory):
    extension = image_type if image_type else 'jpg'
    file_name = str(uuid.uuid4().hex) + "." + extension
    save_path = os.path.join(save_directory, file_name)
    with open(save_path, 'wb+') as image_file:
        image_file.write(raw_image)


#General download method for images
def download_images_to_dir(images, save_directory, num_images):
    for i, (url, image_type) in enumerate(images):
        try:
            logger.info("Making request (%d/%d): %s", i, num_images, url)
            raw_image = get_raw_image(url)
            save_image(raw_image, image_type, save_directory)
        except Exception as e:
            logger.exception(e)


#Method that runs the query
def run(query, save_directory, num_images=100):
    query = '+'.join(query.split())
    logger.info("Extracting image links")
    images = extract_images(query, num_images)
    logger.info("Downloading images")
    download_images_to_dir(images, save_directory, num_images)
    logger.info("Finished")


def main():
    parser = argparse.ArgumentParser(description='Scrape Google images')
    parser.add_argument('-s', '--search', default='tennis ball', type=str, help='search term')
    parser.add_argument('-n', '--num_images', default=1, type=int, help='num images to save')
    parser.add_argument('-d', '--directory', default='./', type=str, help='save directory')
    args = parser.parse_args()
    run(args.search, args.directory, args.num_images)


if __name__ == '__main__':
    main()