import numpy as np
import pandas as pd 
from sqlalchemy import create_engine
from sklearn.metrics.pairwise import cosine_similarity
from cosine_reranker import CosineReranker


class Reranker:
    def __init__(self, csv_file_path) -> None:
        self.df = pd.read_csv(csv_file_path)
        
        self.engine = create_engine('sqlite:///houses.db')
        self.cosineReranker = CosineReranker()


    def rank_by_profit(self, ids):
            pass 


    def rank_by_house_prices(self, ids, ascending=True):
        # print(f"Ranking by house prices...")
        filtered_df = self.df[self.df['ID'].isin(ids)]                               # Filter the DataFrame to include only the rows with the specified IDs
        sorted_df = filtered_df.sort_values(by='price', ascending=ascending)         # Sort the filtered DataFrame by the 'price' column in ascending order
        return {'IDs': sorted_df['ID'].values}


    def rank_by_living_space(self, ids, ascending=True):
        print(f"Ranking by living space...")
        filtered_df = self.df[self.df['ID'].isin(ids)]                               # Filter the DataFrame to include only the rows with the specified IDs
        sorted_df = filtered_df.sort_values(by='living_space', ascending=ascending)  # Sort the filtered DataFrame by the 'price' column in ascending order
        return {'IDs': sorted_df['ID'].values}
    

    def promote_company_x(self, ids, company_name='zillow'):
        print(f"Promoting company: {company_name}...")
        df = self.df[self.df['property_url'].str.contains(company_name, case=False, na=False)]          # matching rows
        all_ids = df['ID'].values 
        promoted_ids = set(all_ids).intersection(set(ids))
        other_ids = set(ids).difference(promoted_ids)
        return {'IDs': list(promoted_ids) + list(other_ids)}


    def set_sql_query(self, query):
        self.sql_query = query


    def rank_by_cosine_similarity(self, clip_query, filtered_ids, clip_image_embeddings) -> str:
        """This tool reranks the CLIP image search returns by using merged SQL and CLIP queries against the CLIP image returns and its paired SQL listing data."""

        sql_records_df = pd.read_sql(self.sql_query, self.engine)
        sql_records = sql_records_df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1).tolist()

        # Get record embeddings
        record_embeddings = self.cosineReranker.record_embedder.embed(sql_records)             
        
        # Get CLIP image embeddings for the fetched records
        ids = sql_records_df['ID'].astype(str).tolist()                                         
        # filtered_ids, clip_image_embeddings = image_embedding_agent_tool(clip_query, ids)
        
        # Filter the record_embeddings to include only those that match the top 5 ids
        filtered_record_embeddings = [record_embeddings[ids.index(id)] for id in filtered_ids]
        filtered_record_embeddings = np.array(filtered_record_embeddings)
        
        # Embed the SQL and CLIP queries
        sql_query_embedding = self.cosineReranker.embed_query(self.sql_query)
        clip_query_embedding = self.cosineReranker.embed_query(clip_query)

        # Concatenate the SQL and CLIP query embeddings
        query_embedding = np.concatenate([sql_query_embedding, clip_query_embedding], axis=1)
        
        # Concatenate the record embeddings with their associated image embeddings
        record_image_embeddings = []
        for record_embedding, id in zip(filtered_record_embeddings, filtered_ids):
            image_embedding = clip_image_embeddings[id]
            record_image_embeddings.append(np.concatenate([record_embedding, image_embedding]))

        record_image_embeddings = np.array(record_image_embeddings)
        
        # Calculate cosine similarity matrix
        cos_sim_matrix = cosine_similarity(query_embedding, record_image_embeddings)

        # Sort the results based on cosine similarity
        sorted_indices = np.argsort(cos_sim_matrix, axis=1)[:, ::-1]
            
        # Map sorted indices to IDs
        sorted_ids = np.array(filtered_ids)[sorted_indices[0]].tolist()
        print("Cosine Similarity Matrix:")
        print(cos_sim_matrix)
        print("Sorted Indices:")
        print(sorted_indices)
        print("Reranked IDs:")
        print(sorted_ids)
        return {
            'cos_sim_matrix': cos_sim_matrix.tolist(),
            'sorted_indices': sorted_indices.tolist(),
            'IDs': sorted_ids
        }


if __name__ == "__main__":
    file_path = 'cleaned_houses_info_with_ID.csv'
    reranker = Reranker(file_path)

    ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print("rank_by_house_prices:", reranker.rank_by_house_prices(ids, False))
    print("rank_by_living_space:", reranker.rank_by_living_space(ids, False))
    print("promote_company_x (zillow):", reranker.promote_company_x(ids, 'zillow'))



'''
(.conda) (base) smsalahuddinkadir@Ss-MacBook-Pro notebooks % python reranker.py
rank_by_house_prices: [ 6  3  1  9  2  4  8  5  7 10]
rank_by_living_space: [ 6  3  2  8  1  5  4  9 10  7]
promote_company_x (zillow): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
'''
