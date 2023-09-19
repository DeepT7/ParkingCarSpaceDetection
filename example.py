from database import get_database
import numpy as np
import pandas as pd

db = get_database()
collection = db['parking']
query = {"nameParking" : "G2_UET_VNU"}
document = collection.find_one(query)

if document:
    # Access the value of the 'spots' field
    spots = document['Slots']
else:
    print("Document not found")

booked_spots = []
for spot in spots:
    if spot['status'] == 2:
        booked_spots.append(int(spot['slot']))
print(booked_spots)