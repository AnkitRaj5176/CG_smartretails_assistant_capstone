import logging

import pandas
import pymongo
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure, OperationFailure

from server.env_config import env_settings

logger = logging.getLogger(__name__)

_mongo_client: MongoClient | None = None


def _get_client() -> MongoClient:
    """Return a cached MongoClient, creating one if needed."""
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(
            env_settings.MONGO_CONNECTION,
            serverSelectionTimeoutMS=5000,
        )
    return _mongo_client


def _get_collection(collection_name: str) -> Collection:
    """Return a MongoDB collection from the configured database."""
    client = _get_client()
    database = client[env_settings.MONGO_DB_NAME]
    return database[collection_name]


def setup_mongo_indexes() -> None:
    """Create indexes on the retail_records collection at application startup."""
    try:
        collection = _get_collection("retail_records")
        collection.create_index(
            [("product_id", pymongo.ASCENDING), ("date", pymongo.ASCENDING)],
            name="idx_product_date", background=True,
        )
        collection.create_index([("store_id", pymongo.ASCENDING)], name="idx_store_id", background=True)
        collection.create_index([("category", pymongo.ASCENDING)], name="idx_category", background=True)
        collection.create_index([("region", pymongo.ASCENDING)], name="idx_region", background=True)
        collection.create_index([("date", pymongo.ASCENDING)], name="idx_date", background=True)
        logger.info("MongoDB indexes created/verified on 'retail_records'.")
    except (ConnectionFailure, OperationFailure) as mongo_error:
        logger.warning("MongoDB index setup failed (server may be unavailable): %s", mongo_error)
    except Exception as unexpected_error:
        logger.warning("Unexpected error during MongoDB setup: %s", unexpected_error)


def save_retail_records(cleaned_dataframe: pandas.DataFrame) -> int:
    """Persist cleaned retail records to MongoDB, replacing any existing data."""
    if cleaned_dataframe.empty:
        logger.warning("save_retail_records called with an empty dataframe — nothing to save.")
        return 0
    try:
        collection = _get_collection("retail_records")
        records_copy = cleaned_dataframe.copy()
        if "date" in records_copy.columns:
            records_copy["date"] = records_copy["date"].dt.to_pydatetime()
        documents = records_copy.to_dict(orient="records")
        collection.delete_many({})
        result = collection.insert_many(documents, ordered=False)
        inserted_count = len(result.inserted_ids)
        logger.info("Saved %d retail records to MongoDB.", inserted_count)
        return inserted_count
    except (ConnectionFailure, OperationFailure) as mongo_error:
        logger.warning("MongoDB write failed (server may be unavailable): %s", mongo_error)
        return 0
    except Exception as unexpected_error:
        logger.warning("Unexpected error saving retail records: %s", unexpected_error)
        return 0
