from typing import Optional, List, Dict, Any
import os
import threading
import re

from fastapi import FastAPI, HTTPException, Path, Query
from pydantic import BaseModel
import pandas as pd


DB_FILES = {
    "2025": {
        "file": "Database.csv",
        "headers": {"seat": "seating_no", "name": "arabic_name", "degree": "total_degree"},
        "max_degree": 320.0
    },
    "2024": {
        "file": "Database2024.csv",
        "headers": {"seat": "رقم الجلوس", "name": "الاسم", "degree": "الدرجة"},
        "max_degree": 410.0
    }
}


DATA_LOCK = threading.Lock()
DATAFRAME_CACHE: Dict[str, Optional[pd.DataFrame]] = {
    year_key: None
    for year_key in DB_FILES.keys()
}


app = FastAPI(title="Multi-year Database API")


class SearchResponseItem(BaseModel):
    seat: Optional[Any]
    name: Optional[Any]
    degree: Optional[Any]


class ResultsPayload(BaseModel):
    total_matches: int
    items: List[SearchResponseItem]


class SearchResponse(BaseModel):
    results: ResultsPayload


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


def detect_columns_from_dataframe_for_year(dataframe: pd.DataFrame, year_key: str) -> Dict[str, Optional[str]]:
    detected_name_column: Optional[str] = None
    detected_seat_column: Optional[str] = None
    detected_degree_column: Optional[str] = None

    column_names = list(dataframe.columns)

    mapping_entry = DB_FILES.get(year_key, {})
    mapping_headers = mapping_entry.get("headers", {})

    arabic_or_custom_name_candidate = str(mapping_headers.get("name", "")).strip().lower()
    arabic_or_custom_seat_candidate = str(mapping_headers.get("seat", "")).strip().lower()
    arabic_or_custom_degree_candidate = str(mapping_headers.get("degree", "")).strip().lower()

    english_name_keywords = ["name", "full_name", "student_name", "student", "candidate", "arabic_name"]
    english_seat_keywords = ["seat", "seat_no", "seat_number", "roll", "roll_no", "roll_number", "seating_no"]
    english_degree_keywords = ["degree", "score", "marks", "mark", "result", "total_degree"]

    for column_name in column_names:
        normalized_column = str(column_name).strip().lower()

        if detected_name_column is None:
            exact_match_custom_name = normalized_column == arabic_or_custom_name_candidate and arabic_or_custom_name_candidate != ""
            contains_custom_name = arabic_or_custom_name_candidate in normalized_column and arabic_or_custom_name_candidate != ""
            contains_arabic_word_name = "اسم" in normalized_column
            exact_english_name = normalized_column in english_name_keywords
            contains_english_name = any(keyword in normalized_column for keyword in english_name_keywords)

            if exact_match_custom_name:
                detected_name_column = column_name
                continue

            if contains_custom_name:
                detected_name_column = column_name
                continue

            if contains_arabic_word_name:
                detected_name_column = column_name
                continue

            if exact_english_name:
                detected_name_column = column_name
                continue

            if contains_english_name:
                detected_name_column = column_name
                continue

        if detected_seat_column is None:
            exact_match_custom_seat = normalized_column == arabic_or_custom_seat_candidate and arabic_or_custom_seat_candidate != ""
            contains_custom_seat = arabic_or_custom_seat_candidate in normalized_column and arabic_or_custom_seat_candidate != ""
            contains_arabic_word_seat = "جلوس" in normalized_column or "رقم" in normalized_column
            exact_english_seat = normalized_column in english_seat_keywords
            contains_english_seat = any(keyword in normalized_column for keyword in english_seat_keywords)

            if exact_match_custom_seat:
                detected_seat_column = column_name
                continue

            if contains_custom_seat:
                detected_seat_column = column_name
                continue

            if contains_arabic_word_seat:
                detected_seat_column = column_name
                continue

            if exact_english_seat:
                detected_seat_column = column_name
                continue

            if contains_english_seat:
                detected_seat_column = column_name
                continue

        if detected_degree_column is None:
            exact_match_custom_degree = normalized_column == arabic_or_custom_degree_candidate and arabic_or_custom_degree_candidate != ""
            contains_custom_degree = arabic_or_custom_degree_candidate in normalized_column and arabic_or_custom_degree_candidate != ""
            contains_arabic_word_degree = "درجة" in normalized_column
            exact_english_degree = normalized_column in english_degree_keywords
            contains_english_degree = any(keyword in normalized_column for keyword in english_degree_keywords)

            if exact_match_custom_degree:
                detected_degree_column = column_name
                continue

            if contains_custom_degree:
                detected_degree_column = column_name
                continue

            if contains_arabic_word_degree:
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
            contains_name_word = "name" in normalized_column or "اسم" in normalized_column
            if contains_name_word:
                detected_name_column = column_name
                break

    if detected_seat_column is None:
        for column_name in column_names:
            normalized_column = str(column_name).strip().lower()
            contains_seat_word = ("seat" in normalized_column) or ("roll" in normalized_column) or ("جلوس" in normalized_column) or ("رقم" in normalized_column)
            if contains_seat_word:
                detected_seat_column = column_name
                break

    if detected_degree_column is None:
        for column_name in column_names:
            normalized_column = str(column_name).strip().lower()
            contains_degree_word = ("deg" in normalized_column) or ("score" in normalized_column) or ("mark" in normalized_column) or ("درجة" in normalized_column)
            if contains_degree_word:
                detected_degree_column = column_name
                break

    return {
        "name_column": detected_name_column,
        "seat_column": detected_seat_column,
        "degree_column": detected_degree_column
    }


def ensure_dataframe_loaded_for_year(year_key: str) -> None:
    if year_key not in DB_FILES:
        raise HTTPException(status_code=404, detail="Year not configured")

    with DATA_LOCK:
        dataframe_is_none = DATAFRAME_CACHE.get(year_key) is None

    if dataframe_is_none:
        csv_path_env = DB_FILES[year_key].get("file")
        if csv_path_env is None:
            raise FileNotFoundError("No file configured for year")

        file_exists_at_path = os.path.exists(csv_path_env)
        if not file_exists_at_path:
            raise FileNotFoundError(csv_path_env)

        loaded_dataframe = read_csv_from_path(csv_path_env)

        with DATA_LOCK:
            DATAFRAME_CACHE[year_key] = loaded_dataframe


@app.on_event("startup")
def load_all_data_on_startup() -> None:
    for year_key in DB_FILES.keys():
        try:
            ensure_dataframe_loaded_for_year(year_key)
        except FileNotFoundError:
            continue


@app.post("/realode")
def reload_all_csvs() -> Dict[str, Any]:
    reloaded_info: Dict[str, Any] = {}

    for year_key, config_entry in DB_FILES.items():
        csv_path_for_year = config_entry.get("file")

        file_exists_at_path = os.path.exists(csv_path_for_year)
        if not file_exists_at_path:
            reloaded_info[year_key] = {"ok": False, "error": "file_not_found"}
            continue

        dataframe_for_year = read_csv_from_path(csv_path_for_year)

        with DATA_LOCK:
            DATAFRAME_CACHE[year_key] = dataframe_for_year

        loaded_rows_count = len(dataframe_for_year.index)
        reloaded_info[year_key] = {"ok": True, "loaded_rows": loaded_rows_count}

    response_payload = {"status": "ok", "details": reloaded_info}

    return response_payload


@app.post("/{year_key}/realode")
def reload_csv_for_year(
    year_key: str = Path(..., description="Year key to reload"),
) -> Dict[str, Any]:
    if year_key not in DB_FILES:
        raise HTTPException(status_code=404, detail="Year not configured")

    csv_path_for_year = DB_FILES[year_key].get("file")

    file_exists_at_path = os.path.exists(csv_path_for_year)
    if not file_exists_at_path:
        raise HTTPException(status_code=404, detail="CSV file not found")

    dataframe_for_year = read_csv_from_path(csv_path_for_year)

    with DATA_LOCK:
        DATAFRAME_CACHE[year_key] = dataframe_for_year

    loaded_rows_count = len(dataframe_for_year.index)

    response_payload = {"status": "ok", "loaded_rows": loaded_rows_count}

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


@app.get("/{year_key}/search/{query}", response_model=SearchResponse)
def search_by_year_and_path(
    year_key: str = Path(..., description="Year key to search"),
    query: str = Path(..., description="Search term (seat number or name)"),
    limit: int = Query(100, ge=1, le=1000)
) -> Dict[str, Any]:
    if year_key not in DB_FILES:
        raise HTTPException(status_code=404, detail="Year not configured")

    ensure_dataframe_loaded_for_year(year_key)

    with DATA_LOCK:
        dataframe_copy = DATAFRAME_CACHE[year_key].copy()

    detected_columns = detect_columns_from_dataframe_for_year(dataframe_copy, year_key)

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

    total_matches_count = len(matched_frame.index)

    if matched_frame.empty:
        items_list: List[Dict[str, Any]] = []
    else:
        items_list = []
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

            items_list.append(item)

    results_payload = {
        "total_matches": total_matches_count,
        "items": items_list
    }

    response_payload = {
        "results": results_payload
    }

    return response_payload
