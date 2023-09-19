from database import get_database
import json 
import pandas as pd
import numpy as np
import time

N = 0
while True:
    start = time.time()
    dbname = get_database()
    collection = dbname['parking']
    query = {"nameParking" : "G2_UET_VNU"}
    document = collection.find_one(query)

    # Check which spots are booked
    if document:
        spots = document['Slots']
    else:
        print("Document not found")

    booked_spots = []
    for spot in spots:
        if spot['status'] == 2:
            booked_spots.append(int(spot['slot']))
    
    try:
        df = pd.read_json('data.json')
    except ValueError as e:
        print(f"Error reading 'data.json': {e}")
    except FileNotFoundError as e:
        print(f"'data.json' not found: {e}")
    emptySpots = np.array(df['empty_spots'])
    counter = int(df['number'][0])
    total_slots = int(df['total_number'][0])
    eS = []
    #print(f"{counter}   {total_slots}")
    for i in range(total_slots):
        one_slot = {'slot' : str(i), 'status' : 1}
        if(i in emptySpots):
            if(i in booked_spots):
                one_slot['status'] = 2
            else:
                one_slot['status'] = 0
        eS.append(one_slot)

    collection.update_one(
        {"nameParking" : "G2_UET_VNU"},
        {"$set" : {"Value_empty_slot" : counter,"Slots": eS},
        "$currentDate" : {"lastModified": True}},
    )
    print(N)
    N = N + 1
    time.sleep(5)
        