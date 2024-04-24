import sys

sys.path.append("../")


from helpers import Errors, ExcelFileReaderHelper, StringDefinitionsHelper

def get_columns(file_name):
    """

    :param file_name: The name of the file being read in

    Gets all the required column names from the excel file that's stored in the data folder.

    """
    efrh = ExcelFileReaderHelper.ExcelFileReaderHelper()
    efrh.read_from_excel(file_path=file_name)

    columns = efrh.get_sheet_columns()

    return columns

def get_data(file_name, file_format, clustered_column):
    """

    :param file_name: The name of the file being read in
    :param file_format: The format of the file being read
    :param clustered_column: The type of column we are clustering by
    :return: One DataFrame for the data being clustered, one for the x values, and one for the y values

    Reads in all of the data from an excel file that's stored in the data folder.

    """
    efrh = ExcelFileReaderHelper.ExcelFileReaderHelper()
    efrh.read_from_excel(file_path=file_name)
    # TODO Specify the sheet name, have it just be a possible option
    if file_format == StringDefinitionsHelper.FILE_FORMAT_ONE:
        print("file format is one")
        hard_df, modu_df, x_df, y_df,hard_mod_df = efrh.read_next_sheet_format1(nulls=True)
        print("File format 1 processed")
    elif file_format == StringDefinitionsHelper.FILE_FORMAT_TWO:
        print("file format is two")
        hard_df, modu_df, x_df, y_df,hard_mod_df = efrh.read_next_sheet_format2(nulls=True)
    elif file_format == StringDefinitionsHelper.FILE_FORMAT_THREE:
        print("file format is three")
        hard_df, modu_df, x_df, y_df, hard_mod_df = efrh.read_next_sheet_format3(clustered_column, nulls=True)
        print("File format 3 processed")
    else:
        raise Errors.InvalidClusteringFileFormat(file_format)
    
    if clustered_column == StringDefinitionsHelper.HARDNESS_LABEL:
        data_df = hard_df
    elif clustered_column == StringDefinitionsHelper.MODULUS_LABEL:
        data_df = modu_df
    elif clustered_column=="Hard_Mod":
        data_df=hard_mod_df
    elif isinstance(clustered_column, list):
        data_df = hard_mod_df
    else:
        print("there was an error")
        raise Errors.InvalidClusteringColumn(clustered_column)
    
    print("x_df: ", x_df)
    print("y_df: ", y_df)

    #Exporting the dataframe to excel in server files
    data_df.to_excel('../server_files/dataframe.xlsx', index=False)
    
    #data_df=hard_mod_df
    return data_df, x_df, y_df
