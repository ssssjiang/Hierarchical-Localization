# shu.song@ninebot.com
import sys
import sqlite3
import numpy as np

IS_PYTHON3 = sys.version_info[0] >= 3

MAX_IMAGE_ID = 2 ** 31 - 1

CREATE_GLOBAL_DESCRIPTOR_TABLE = """CREATE TABLE IF NOT EXISTS global_descriptor (
    image_id INTEGER PRIMARY KEY NOT NULL,
    name TEXT    NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_LOCAL_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS local_descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    name TEXT    NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    pixel BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    name TEXT    NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_ALL = "; ".join([
    CREATE_LOCAL_DESCRIPTORS_TABLE,
    CREATE_KEYPOINTS_TABLE,
    CREATE_GLOBAL_DESCRIPTOR_TABLE
])


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) / MAX_IMAGE_ID
    return image_id1, image_id2


def array_to_blob(array):
    if IS_PYTHON3:
        return array.tobytes()
    else:
        return np.getbuffer(array)


def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)


class HFNetDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(str(database_path), factory=HFNetDatabase)

    def __init__(self, *args, **kwargs):
        super(HFNetDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_local_descriptors_table = \
            lambda: self.executescript(CREATE_LOCAL_DESCRIPTORS_TABLE)
        self.create_global_descriptors_table = \
            lambda: self.executescript(CREATE_GLOBAL_DESCRIPTOR_TABLE)

    def add_keypoints(self, image_id, image_name, keypoints):
        assert (len(keypoints.shape) == 2)
        assert (keypoints.shape[1] in [3, 4, 6])

        keypoints = np.asarray(keypoints, np.float32)
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?, ?)",
            (image_id,) + (image_name, ) + keypoints.shape + (array_to_blob(keypoints),))

    def add_local_descriptors(self, image_id, image_name, local_descriptors):
        local_descriptors = np.ascontiguousarray(local_descriptors, np.np.float32)
        self.execute(
            "INSERT INTO local_descriptors VALUES (?, ?, ?, ?)",
            (image_id,) + (image_name, ) + local_descriptors.shape + (array_to_blob(local_descriptors),))

    def add_global_descriptors(self, image_id, image_name, global_descriptors):
        global_descriptors = np.ascontiguousarray(global_descriptors, np.np.float32)
        self.execute(
            "INSERT INTO global_descriptors VALUES (?, ?, ?, ?)",
            (image_id,) + (image_name, ) + global_descriptors.shape + (array_to_blob(global_descriptors),))

    def read_keypoints_from_image_name(self, image_name):
        cursor = self.execute(
            'SELECT pixel FROM keypoints WHERE name=?;',  (image_name,))
        keypoints = cursor.fetchone()
        if keypoints is None or keypoints[0] is None:
            return None
        keypoints = np.fromstring(keypoints[0], dtype=np.float32).reshape(-1, 3)
        return keypoints

    def read_local_descriptors_from_image_name(self, image_name):
        cursor = self.execute(
            'SELECT data FROM local_descriptors WHERE name=?;',  (image_name,))
        local_descriptors = cursor.fetchone()
        if local_descriptors is None or local_descriptors[0] is None:
            return None
        local_descriptors = np.fromstring(local_descriptors[0], dtype=np.float32).reshape(-1, 256)
        return local_descriptors


def example_usage():
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--database_path", default="database.db")
    args = parser.parse_args()

    if os.path.exists(args.database_path):
        print("ERROR: database path already exists -- will not modify it.")
        return

    hfnet_database = args.database_path

    hfnet_connection = sqlite3.connect(hfnet_database)
    hfnet_cursor = hfnet_connection.cursor()

    # Read for specific image_id
    image_id = 0
    hfnet_cursor.execute(
        'SELECT pixel FROM keypoints WHERE image_id=?;',  (image_id,))
    row = next(hfnet_cursor)
    keypoints = np.fromstring(row[0], dtype=np.float32).reshape(-1, 2)

    hfnet_cursor.execute(
        'SELECT data FROM local_descriptors WHERE image_id=?;', (image_id,))
    row = next(hfnet_cursor)
    local_descriptors = np.fromstring(row[0], dtype=np.float32).reshape(-1, 256)

    hfnet_cursor.execute(
        'SELECT data FROM global_descriptor WHERE image_id=?;', (image_id,))
    row = next(hfnet_cursor)
    global_descriptor = np.fromstring(row[0], dtype=np.float32).reshape(-1, 4096)


    # Read all keypoints.
    keypoints = dict(
        (image_id,
         blob_to_array(data, np.float32, (-1, 2)))
        for image_id, data in hfnet_cursor.execute("SELECT image_id, pixel FROM keypoints")
    )
    local_descriptors = dict(
        (image_id,
         blob_to_array(data, np.float32, (-1, 256)))
        for image_id, data in hfnet_cursor.execute("SELECT image_id, data FROM local_descriptors")
    )
    global_descriptor = dict(
        (image_id,
         blob_to_array(data, np.float32, (-1, 4096)))
        for image_id, data in hfnet_cursor.execute("SELECT image_id, data FROM global_descriptor")
    )

    # Clean up.

    hfnet_connection.close()

    if os.path.exists(args.database_path):
        os.remove(args.database_path)


if __name__ == "__main__":
    example_usage()
