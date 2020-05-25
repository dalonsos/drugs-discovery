import sqlite3
from sqlite3 import Error


def create_connection(db_file):
    """
    create a database connection to the SQLite database
    specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    print("Connecting to database ...")
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return conn


def run_query(conn, query):
    """
    runs a query
    :param conn: database connection
    :param query: query
    :return: list with rows
    """
    print("Running query ...")
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    rows = [row[0] for row in rows]

    return rows


