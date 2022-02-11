from chapter_2 import input_data

if __name__ == "__main__":
    input_data.fetch_housing_data()
    input_df = input_data.load_housing_data()
