"""
Get SMILES that annotated nM activities
"""

from data_preprocessing.db_utils import *


DATABASE = "../chembl_25_sqlite/chembl_25.db"
OUTPUT_SMI_FILE = "../data/chembl_25.smi"
QUERY = """
    SELECT
      DISTINCT canonical_smiles
    FROM
      compound_structures
    WHERE
      molregno IN (
        SELECT
          DISTINCT molregno
        FROM
          activities
        WHERE
          standard_type IN ("Kd", "Ki", "Kb", "IC50", "EC50")
          AND standard_units = "nM"
          AND standard_value < 1000
          AND standard_relation IN ("<", "<<", "<=", "=")
        INTERSECT
        SELECT
          molregno
        FROM
          molecule_dictionary
        WHERE
          molecule_type = "Small molecule"
      );
    """


if __name__ == '__main__':
    # create a database connection
    conn = create_connection(DATABASE)
    with conn:
        # run query
        rows = run_query(conn, QUERY)
        file = open(OUTPUT_SMI_FILE, "w")
        for row in rows:
            file.write(row)
            file.write("\n")
        file.close()

