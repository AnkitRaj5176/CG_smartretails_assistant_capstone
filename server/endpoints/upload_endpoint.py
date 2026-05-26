import logging
import os
import shutil

from fastapi import APIRouter, HTTPException, UploadFile, File, status

from server.forecasting.csv_reader import read_retail_csv, check_mandatory_columns
from server.forecasting.record_cleaner import clean_retail_records
from server.infra.mongo_store import save_retail_records

logger = logging.getLogger(__name__)

upload_router = APIRouter(tags=["A. Data Ingestion"])

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_SALES_STORE_PATH: str = os.path.join(_project_root, "sales_store")
_RETAIL_RECORDS_PATH: str = os.path.join(_SALES_STORE_PATH, "retail_records.csv")


@upload_router.post("/api/data/upload", status_code=status.HTTP_200_OK)
async def upload_retail_csv(file: UploadFile = File(...)) -> dict:
    """Accept a CSV upload, validate, clean, and persist the retail records."""
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .csv files are accepted.",
        )

    os.makedirs(_SALES_STORE_PATH, exist_ok=True)

    temp_file_path = _RETAIL_RECORDS_PATH + ".tmp"
    try:
        with open(temp_file_path, "wb") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
    except OSError as write_error:
        logger.warning("Failed to write uploaded file: %s", write_error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file.",
        )

    try:
        raw_dataframe = read_retail_csv(temp_file_path)
    except Exception as read_error:
        os.remove(temp_file_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read CSV: {read_error}",
        )

    original_row_count = len(raw_dataframe)

    try:
        check_mandatory_columns(raw_dataframe)
    except ValueError as column_error:
        os.remove(temp_file_path)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(column_error),
        )

    try:
        cleaned_dataframe = clean_retail_records(raw_dataframe)
    except Exception as clean_error:
        os.remove(temp_file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Data cleaning failed: {clean_error}",
        )

    cleaned_row_count = len(cleaned_dataframe)

    try:
        cleaned_dataframe.to_csv(_RETAIL_RECORDS_PATH, index=False)
        os.remove(temp_file_path)
    except OSError as save_error:
        logger.warning("Failed to save cleaned CSV: %s", save_error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist cleaned data.",
        )

    save_retail_records(cleaned_dataframe)

    logger.info("Upload complete — original=%d, cleaned=%d", original_row_count, cleaned_row_count)

    return {
        "status_message": "File uploaded and processed successfully.",
        "original_row_count": original_row_count,
        "cleaned_row_count": cleaned_row_count,
    }
