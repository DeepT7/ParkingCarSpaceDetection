from pymongo import MongoClient

def get_database():
    CONNECT_STRING = "mongodb+srv://sparking:Az123456@dbs.bgpecdq.mongodb.net/"
    client = MongoClient(CONNECT_STRING)

    return client['ATH_UET']

if __name__ == "__main__":
    dbname = get_database()