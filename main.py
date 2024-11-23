from loadData import DataLoader

def main():
    loader = DataLoader()
    loader.load_csv("data/features_3_sec.csv")
    dataframe = loader.load_csv("data/features_30_sec.csv")
    print(dataframe.head())

if __name__ == "__main__":
    main()