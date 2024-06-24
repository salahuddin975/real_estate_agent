import pandas as pd
from sqlalchemy import create_engine, Column, Integer, Float, String, MetaData, Table


class SQLAgent:
    def __init__(self, csv_file_path):
        # csv_file_path = "houses_dataset/cleaned_houses_info_with_ID.csv"
        self.setup_database(csv_file_path)


    def setup_database(self, csv_file_path):
        df = pd.read_csv(csv_file_path)

        self.engine = create_engine('sqlite:///houses.db')           # Create SQLAlchemy engine for SQLite
        metadata = MetaData()                                   # Define the table schema

        house_listings = Table(
            'houses', metadata,
            Column('ID', Integer, primary_key=True),
            Column('bedrooms', Integer),
            Column('bathrooms', Integer),
            Column('living_space', Float),
            Column('address', String),
            Column('city', String),
            Column('state', String),
            Column('zipcode', Integer),
            Column('latitude', Float),
            Column('longitude', Float),
            Column('property_url', String),
            Column('price', Float)
        )

        metadata.create_all(self.engine)                             # Create the table in the database
        df.to_sql('house_listings', self.engine, if_exists='replace', index=False)            # Insert data into the table


    def execute_query(self, sql_query):
        result = pd.read_sql(sql_query, self.engine)
        return result['ID'].values  if len(result['ID']) > 0 else list(range(1, 535))
    

if __name__ == "__main__":
    csv_file_path = "houses_dataset/cleaned_houses_info_with_ID.csv"
    sqlagent = SQLAgent(csv_file_path)

    query = "SELECT * FROM house_listings WHERE bedrooms <= 3 AND bathrooms <= 3 AND city = 'Yuba City' AND state = 'CA' AND price <= 600000;"
    res = sqlagent.execute_query(query)
    print(res)
