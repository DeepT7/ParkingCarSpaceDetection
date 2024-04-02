#!/usr/bin/env python
import os
import sys
import cv2
import csv
import json
import time
import datetime
import psycopg2
import argparse
import numpy as np
import torch 

from shapely import wkb
from shapely.wkb import loads
from shapely.geometry import box, Polygon, MultiPolygon, Point
from scipy.spatial import cKDTree

from utility.config import config
from utility import detection


file_path = "./data/coordinates.csv"

def main(args):
    global conn, cur

    # connect to PostgreSQL database
    # db_params = config(filename=CONFIG_INI_FILE)
    # conn = psycopg2.connect(**db_params)
    # cur = conn.cursor()

    # load detection model
    # model = detection.load_inference_resnet50()
    model = torch.hub.load('ultralytics/yolov5', 'custom', 'trained_model/best9.pt')


    # fetch areas that will be analyzed
    spots = fetch_parking_spots()

    # get video data
    vcap = cv2.VideoCapture(args.video_file)
    fps = int(vcap.get(cv2.CAP_PROP_FPS))
    frame_counter = 0
    detection_interval = fps * time2seconds(args.time_interval)

    # start analyzing parking lot
    while vcap.isOpened():
        ret, frame = vcap.read()
        key = cv2.waitKey(fps) & 0xFF

        # end of video or user exit
        if not ret or key == ord("q"):
            print("Video stopped")
            break

        # check if parking spots are occupied every nth frame
        if frame_counter % detection_interval == 0:
            # set occupancy for each spot to false
            #reset_occupancy(args.cam_ids)

            # detect which spots are occupied
            bboxes = detection.detect_cars(
                model, frame, [3, 4], threshold=0.5)
            occupied_spots = fetch_occupied_spots(spots, bboxes)

            # update occupancy in table for each spot
            is_occupied = update_occupancy(spots, occupied_spots)
            
        # check if spot_time > time_threshold
        frame_counter += 1

        # display video
        frame = display(frame, is_occupied, spots)
        cv2.imshow("parking lot", frame)
        cv2.waitKey(25)

    # reset and close connections
    vcap.release()


def display(frame, occupied, spots):
    mask = frame.copy()
    empty_spots = []
    i =  0
    # draw parking spots
    for spot, is_occupied in zip(spots, occupied):
        if is_occupied:
            color = (0, 0, 255)
        else:
            empty_spots.append(i)
            color = (0, 255, 0)
        coords = np.array(spot.exterior.coords, dtype="int")
        cv2.polylines(mask, [coords], True, color, thickness = 2)
        i += 1
    
    # make colors more transparent
    mask = cv2.addWeighted(mask, 0.6, frame, 0.4, 0)
    data = {
            'name' : "Parking 3",
            'total_number' : len(spots),
            'number': len(empty_spots),
            'empty_spots': empty_spots,
            }
    json_data = json.dumps(data)
    with open('data.json', 'w') as f:
        f.write(json_data)
    return mask


def time2seconds(time_interval):
    x = time.strptime(time_interval, "%H:%M:%S")
    seconds = datetime.timedelta(
        hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds()
    return int(seconds)


def reset_occupancy(cam_ids):
    query = """UPDATE spots
               SET is_occupied = false
               WHERE camera_id = ANY(%s);"""
    cur.execute(query, (cam_ids,))
    conn.commit()


def fetch_parking_spots():
    spots = []

    with open(file_path, 'r') as csvfile:
        for line in csvfile:
            line = line.strip().replace('"', '')
            coordinates_pairs = line.split('),(')

            # Convert the coordinate pairs to tuples
            sublist = [tuple(map(int, pair.strip('()').split(','))) for pair in coordinates_pairs if pair]
            spots.append(sublist)
    
    polygon_spots = [Polygon(vertices) for vertices in spots]
    # spots is a Shapely object
    return polygon_spots


def fetch_occupied_spots(spots, candidates):
    occupied_spots = []
    centroids = detection.fetch_centroids(candidates)
    tree = cKDTree(centroids)

    # get nearest detected car for each spot
    for spot in spots:
        #dist, idx = tree.query(spot.centroid, k=1)
        centroid = spot.centroid
        dist, idx = tree.query([centroid.x, centroid.y], k = 1)
        candidate = box(*candidates[idx])
        # if overlap > threshold then spot is occupied
        if detection.is_occupied(spot, candidate):
            occupied_spots.append((spot.wkb_hex,))
    
    return occupied_spots

def update_occupancy(spots, occupied_spots):
    # Convert the occupied_spots tuples into Point geometries
    #occupied_geometry = [Point(coords) for coords in occupied_spots]

    occupancy_status = []

    for spot in spots:
        is_occupied = any(spot.intersects(loads(occupied_spot[0])) for occupied_spot in occupied_spots)
        #is_occupied = any(occupied_spot.intersects(spot) for occupied_spot in occupied_spots)
        #is_occupied = any(spot.intersects(occupied_spot) for occupied_spot in occupied_spots )
        occupancy_status.append(is_occupied)

    return occupancy_status
# def reset_occupancy(cam_ids):
#     query = """UPDATE spots
#                SET is_occupied = false
#                WHERE camera_id = ANY(%s);"""
#     cur.execute(query, (cam_ids,))
#     conn.commit()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    # TODO: Data is supposed to be fetched from
    # live feed, not from a video file.
    parser.add_argument("video_file", type=str,
                        help="path/to/video.mp4")
    parser.add_argument("--time_interval", "-t", type=str, default="00:00:05",
                        help="do object detection every time interval")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
