from typing import Optional, List, Dict, Any
import os
import threading
import re

from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel
import pandas as pd


DATA_LOCK = threading.Lock()
DATAFRAME_IN_MEMORY: Optional[pd.DataFrame] = None

CSV_FILE_DEFAULT = "Database2024_Stage_New_Search.csv"
HEADERS_MAPPING = {"seat": "رقم الجلوس", "name": "الاسم", "degree": "الدرجة"}
MAX_DEGREE_VALUE = 410.0

app = FastAPI(title="Database2024 API")


class SearchResponseItem(BaseModel):
    seat: Optional[Any]
    name: Optional[Any]
    degree: Optional[Any]


class SearchResponse(BaseModel):
    meta: Dict[str, Any]
    results: List[SearchResponseItem]


def read_csv_from_path(csv_path: str) -> pd.DataFrame:
    try:
        dataframe_loaded = pd.read_csv(
            csv_path,
            encoding="utf-8-sig",
            header=0,
            dtype=str,
            low_memory=False,
            keep_default_na=False
        )
    except Exception:
        dataframe_loaded = pd.read_csv(
            csv_path,
            header=0,
            dtype=str,
            low_memory=False,
            keep_default_na=False
        )

    column_names = list(dataframe_loaded.columns)

    looks_like_no_header = True

    for column_name in column_names:
        if column_name is None:
            continue

        normalized_column = str(column_name).strip().lower()

        if normalized_column == "":
            continue

        if not normalized_column.startswith("unnamed"):
            looks_like_no_header = False
            break

    if looks_like_no_header:
        first_row_values = dataframe_loaded.iloc[0].astype(str).tolist()

        dataframe_with_row_headers = pd.read_csv(
            csv_path,
            header=None,
            dtype=str,
            low_memory=False,
            keep_default_na=False
        )

        new_header = first_row_values

        dataframe_with_row_headers.columns = new_header

        dataframe_loaded = dataframe_with_row_headers.iloc[1:].reset_index(drop=True)

    return dataframe_loaded


def detect_columns_from_dataframe(dataframe: pd.DataFrame) -> Dict[str, Optional[str]]:
    detected_name_column: Optional[str] = None
    detected_seat_column: Optional[str] = None
    detected_degree_column: Optional[str] = None

    column_names = list(dataframe.columns)

    arabic_name_candidate = str(HEADERS_MAPPING.get("name", "")).strip().lower()
    arabic_seat_candidate = str(HEADERS_MAPPING.get("seat", "")).strip().lower()
    arabic_degree_candidate = str(HEADERS_MAPPING.get("degree", "")).strip().lower()

    english_name_keywords = ["name", "full_name", "student_name", "student", "candidate"]
    english_seat_keywords = ["seat", "seat_no", "seat_number", "roll", "roll_no", "roll_number"]
    english_degree_keywords = ["degree", "score", "marks", "mark", "result"]

    for column_name in column_names:
        normalized_column = str(column_name).strip().lower()

        if detected_name_column is None:
            exact_match_arabic_name = normalized_column == arabic_name_candidate and arabic_name_candidate != ""
            contains_arabic_name = arabic_name_candidate in normalized_column and arabic_name_candidate != ""
            exact_english_name = normalized_column in english_name_keywords
            contains_english_name = any(keyword in normalized_column for keyword in english_name_keywords)

            if exact_match_arabic_name:
                detected_name_column = column_name
                continue

            if contains_arabic_name:
                detected_name_column = column_name
                continue

            if exact_english_name:
                detected_name_column = column_name
                continue

            if contains_english_name:
                detected_name_column = column_name
                continue

        if detected_seat_column is None:
            exact_match_arabic_seat = normalized_column == arabic_seat_candidate and arabic_seat_candidate != ""
            contains_arabic_seat = arabic_seat_candidate in normalized_column and arabic_seat_candidate != ""
            exact_english_seat = normalized_column in english_seat_keywords
            contains_english_seat = any(keyword in normalized_column for keyword in english_seat_keywords)

            if exact_match_arabic_seat:
                detected_seat_column = column_name
                continue

            if contains_arabic_seat:
                detected_seat_column = column_name
                continue

            if exact_english_seat:
                detected_seat_column = column_name
                continue

            if contains_english_seat:
                detected_seat_column = column_name
                continue

        if detected_degree_column is None:
            exact_match_arabic_degree = normalized_column == arabic_degree_candidate and arabic_degree_candidate != ""
            contains_arabic_degree = arabic_degree_candidate in normalized_column and arabic_degree_candidate != ""
            exact_english_degree = normalized_column in english_degree_keywords
            contains_english_degree = any(keyword in normalized_column for keyword in english_degree_keywords)

            if exact_match_arabic_degree:
                detected_degree_column = column_name
                continue

            if contains_arabic_degree:
                detected_degree_column = column_name
                continue

            if exact_english_degree:
                detected_degree_column = column_name
                continue

            if contains_english_degree:
                detected_degree_column = column_name
                continue

    if detected_name_column is None:
        for column_name in column_names:
            normalized_column = str(column_name).strip().lower()
            if "name" in normalized_column or "اسم" in normalized_column:
                detected_name_column = column_name
                break

    if detected_seat_column is None:
        for column_name in column_names:
            normalized_column = str(column_name).strip().lower()
            if "seat" in normalized_column or "roll" in normalized_column or "جلوس" in normalized_column or "رقم" in normalized_column:
                detected_seat_column = column_name
                break

    if detected_degree_column is None:
        for column_name in column_names:
            normalized_column = str(column_name).strip().lower()
            if "deg" in normalized_column or "score" in normalized_column or "mark" in normalized_column or "درجة" in normalized_column:
                detected_degree_column = column_name
                break

    return {
        "name_column": detected_name_column,
        "seat_column": detected_seat_column,
        "degree_column": detected_degree_column
    }


def ensure_dataframe_loaded() -> None:
    global DATAFRAME_IN_MEMORY

    with DATA_LOCK:
        dataframe_is_none = DATAFRAME_IN_MEMORY is None

    if dataframe_is_none:
        csv_path_env = os.environ.get("DATABASE_CSV_PATH")
        if csv_path_env is None:
            csv_path_env = CSV_FILE_DEFAULT

        file_exists_at_path = os.path.exists(csv_path_env)
        if not file_exists_at_path:
            raise FileNotFoundError(csv_path_env)

        loaded_dataframe = read_csv_from_path(csv_path_env)

        with DATA_LOCK:
            DATAFRAME_IN_MEMORY = loaded_dataframe


@app.on_event("startup")
def load_data_on_startup() -> None:
    try:
        ensure_dataframe_loaded()
    except FileNotFoundError:
        pass


@app.post("/realode")
def reload_csv_into_memory() -> Dict[str, Any]:
    global DATAFRAME_IN_MEMORY

    csv_path_env = os.environ.get("DATABASE_CSV_PATH")
    if csv_path_env is None:
        csv_path_env = CSV_FILE_DEFAULT

    file_exists_at_path = os.path.exists(csv_path_env)
    if not file_exists_at_path:
        raise HTTPException(status_code=404, detail="CSV file not found")

    new_dataframe = read_csv_from_path(csv_path_env)

    with DATA_LOCK:
        DATAFRAME_IN_MEMORY = new_dataframe

    loaded_rows_count = len(DATAFRAME_IN_MEMORY.index)

    response_payload = {
        "status": "ok",
        "message": "CSV reloaded into memory",
        "loaded_rows": loaded_rows_count
    }

    return response_payload


def is_string_all_digits(input_string: str) -> bool:
    stripped_string = input_string.strip()
    match_result = re.fullmatch(r"\d+", stripped_string)
    is_digits = match_result is not None
    return is_digits


def perform_name_search(dataframe: pd.DataFrame, name_column: str, search_term: str, result_limit: int) -> pd.DataFrame:
    series_values = dataframe[name_column].astype(str)
    boolean_mask = series_values.str.contains(search_term, case=False, na=False)
    matched_rows = dataframe[boolean_mask]
    limited_rows = matched_rows.head(result_limit)
    return limited_rows


def perform_seat_search(dataframe: pd.DataFrame, seat_column: str, search_term: str, result_limit: int) -> pd.DataFrame:
    series_values = dataframe[seat_column].astype(str)
    stripped_series_values = series_values.str.strip()
    exact_match_mask = stripped_series_values == search_term
    matched_by_exact = dataframe[exact_match_mask]

    if len(matched_by_exact.index) > 0:
        limited_rows = matched_by_exact.head(result_limit)
        return limited_rows

    contains_mask = stripped_series_values.str.contains(search_term, case=False, na=False)
    matched_by_contains = dataframe[contains_mask]
    limited_rows = matched_by_contains.head(result_limit)
    return limited_rows


@app.get("/2024/search/{query}", response_model=SearchResponse)
def search_2024_by_path(
    query: str = Path(..., description="Search term (seat number or name)"),
    limit: int = 100
) -> Dict[str, Any]:
    ensure_dataframe_loaded()

    with DATA_LOCK:
        dataframe_copy = DATAFRAME_IN_MEMORY.copy()

    detected_columns = detect_columns_from_dataframe(dataframe_copy)

    name_column_detected = detected_columns.get("name_column")
    seat_column_detected = detected_columns.get("seat_column")
    degree_column_detected = detected_columns.get("degree_column")

    if name_column_detected is None and seat_column_detected is None:
        raise HTTPException(status_code=500, detail="CSV does not contain identifiable name or seat columns")

    query_normalized = query.strip()

    is_query_numeric = is_string_all_digits(query_normalized)

    if is_query_numeric:
        if seat_column_detected is None:
            raise HTTPException(status_code=400, detail="Seat column not found in CSV")

        matched_frame = perform_seat_search(
            dataframe=dataframe_copy,
            seat_column=seat_column_detected,
            search_term=query_normalized,
            result_limit=limit
        )
    else:
        if name_column_detected is None:
            raise HTTPException(status_code=400, detail="Name column not found in CSV")

        matched_frame = perform_name_search(
            dataframe=dataframe_copy,
            name_column=name_column_detected,
            search_term=query_normalized,
            result_limit=limit
        )

    if matched_frame.empty:
        results_list: List[Dict[str, Any]] = []
    else:
        results_list = []
        for _, row in matched_frame.iterrows():
            seat_value = None
            name_value = None
            degree_value = None

            if seat_column_detected in row.index:
                seat_value = row[seat_column_detected]

            if name_column_detected in row.index:
                name_value = row[name_column_detected]

            if degree_column_detected in row.index:
                degree_value = row[degree_column_detected]

            item = {
                "seat": seat_value,
                "name": name_value,
                "degree": degree_value
            }

            results_list.append(item)

    total_matches_count = len(matched_frame.index)

    response_payload = {
        "meta": {
            "headers": HEADERS_MAPPING,
            "max_degree": MAX_DEGREE_VALUE,
            "columns_detected": {
                "name_column": name_column_detected,
                "seat_column": seat_column_detected,
                "degree_column": degree_column_detected
            },
            "total_matches": total_matches_count
        },
        "results": results_list
    }

    return response_payload
