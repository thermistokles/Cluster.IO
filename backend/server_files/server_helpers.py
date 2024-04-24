from helpers import MainCallable, StringDefinitionsHelper
from helpers import DataCollectionHelper

def get_column_names1(file_name):
    """
    :param file_name: File from which column names should be extracted
    :return: Array of column names
    """
    columns = DataCollectionHelper.get_columns(file_name=file_name)
    return columns

async def cluster_helper(method, clustering_details,file_name,clusterDataOn):
    """
    Runs clustering methods with a default file
    :param method: the clustering method to run
    :param clustering_details: the parameters for the clustering method
    :return: A response results and/or errors
    """
    scores = []
    try:
        scores = await MainCallable.execute(method, clustering_details, file_name, StringDefinitionsHelper.FILE_FORMAT_THREE,
                                   clustered_column=clusterDataOn)
    except Exception as e:
        error = e.args
        response = createResponse(e, "","","")
        return response, scores
    else:
        rawData = ""
        clusteredData = ""
        clustersFractions = ""
        response = createResponse("", rawData, clusteredData, clustersFractions)
        return response, scores


def createResponse(error, rawData, clusteredData,clustersFractions ):
    """
    Formats results and errors for sending as a response via the API
    :param error: errors
    :param result: results
    :return: a dict of results and errors
    """
    return {"error": error, "rawData": rawData, "clusteredData":clusteredData, "clustersFractions":clustersFractions}
