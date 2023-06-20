def to_kaggle_csv(dataframe, csv_file_path: str = r"./kaggle_submission.csv",
                  molecule_id_column = "molecule_id",
                  molecule_prediction_column = "prediction"):
    result = dataframe[[molecule_id_column, molecule_prediction_column]].copy(deep=True)
    result = result.drop_duplicates(subset=[molecule_id_column]).reset_index(drop=True)
    result = result.sort_values(by=molecule_id_column).reset_index(drop=True)
    result.rename(columns={molecule_prediction_column: "prediction"}, inplace=True)
    result[["prediction"]].to_csv(csv_file_path, index=False)