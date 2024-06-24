import pandas as pd 


class Reranker:
    def __init__(self, csv_file_path) -> None:
        self.df = pd.read_csv(csv_file_path)


    def rank_by_profit(self, ids):
            pass 


    def rank_by_house_prices(self, ids, ascending=True):
        # print(f"Ranking by house prices...")
        filtered_df = self.df[self.df['ID'].isin(ids)]                               # Filter the DataFrame to include only the rows with the specified IDs
        sorted_df = filtered_df.sort_values(by='price', ascending=ascending)         # Sort the filtered DataFrame by the 'price' column in ascending order
        return sorted_df['ID'].values


    def rank_by_living_space(self, ids, ascending=True):
        print(f"Ranking by living space...")
        filtered_df = self.df[self.df['ID'].isin(ids)]                               # Filter the DataFrame to include only the rows with the specified IDs
        sorted_df = filtered_df.sort_values(by='living_space', ascending=ascending)  # Sort the filtered DataFrame by the 'price' column in ascending order
        return sorted_df['ID'].values
    

    def promote_company_x(self, ids, company_name='zillow'):
        print(f"Promoting company: {company_name}...")
        df = self.df[self.df['property_url'].str.contains(company_name, case=False, na=False)]          # matching rows
        all_ids = df['ID'].values 
        promoted_ids = set(all_ids).intersection(set(ids))
        other_ids = set(ids).difference(promoted_ids)
        return list(promoted_ids) + list(other_ids) 
    

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
