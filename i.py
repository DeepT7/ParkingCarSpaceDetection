import os
import sys
import csv
import cv2
import logging
import psycopg2
import argparse

from shapely.geometry import Polygon
from utility import config as my_config
from utility import events



def main(video_file):
    # Fetch video metadata
    cap = cv2.VideoCapture(video_file)
    res_x = int(cap.get(3))
    res_y = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    ret, frame = cap.read()
    coords = events.select_area()

    with open('./data/coordinates.csv', 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(coords)

    # Populate entries for spots table
    for pts in coords:
        polygon = events.sort2cyclic(pts)
        spot = Polygon(polygon)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main("videos/parking_lot_1.mp4")
