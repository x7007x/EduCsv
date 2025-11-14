from typing import Optional, List, Dict, Any
import os
import threading
import re

from fastapi import FastAPI, HTTPException, Path, Query
from pydantic import BaseModel
import pandas as pd


DB_FILES = {
    "2025": {
        "file": "Database2025.csv",
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


app = FastAPI(title="Simple CSV Search API")


class SearchResponseItem(BaseModel):
    seat: Optional[Any]
    name: Optional[Any]
    degree: Optional[Any]
    percentage: Optional[float]


class ResultsPayload(BaseModel):
    total_matches: int
    items: List[SearchResponseItem]


class SearchResponse(BaseModel):
    results: ResultsPayload


def read_csv(csv_path: str) -> pd.DataFrame:
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


def load_dataframe_for_year(year_key: str) -> None:
    if year_key not in DB_FILES:
        raise HTTPException(status_code=404, detail="Year not configured")

    config_entry = DB_FILES[year_key]

    csv_path = config_entry.get("file")

    if csv_path is None:
        raise HTTPException(status_code=500, detail="No file configured for year")

    file_exists_at_path = os.path.exists(csv_path)

    if not file_exists_at_path:
        raise HTTPException(status_code=404, detail="CSV file not found")

    dataframe_loaded = read_csv(csv_path)

    with DATA_LOCK:
        DATAFRAME_CACHE[year_key] = dataframe_loaded


def ensure_loaded_for_year(year_key: str) -> None:
    with DATA_LOCK:
        dataframe_is_none = DATAFRAME_CACHE.get(year_key) is None

    if dataframe_is_none:
        load_dataframe_for_year(year_key)


def detect_columns(year_key: str, dataframe: pd.DataFrame) -> Dict[str, Optional[str]]:
    mapping_entry = DB_FILES.get(year_key, {})
    mapping_headers = mapping_entry.get("headers", {})

    expected_seat_name = str(mapping_headers.get("seat", "")).strip().lower()
    expected_name_name = str(mapping_headers.get("name", "")).strip().lower()
    expected_degree_name = str(mapping_headers.get("degree", "")).strip().lower()

    detected_seat_column: Optional[str] = None
    detected_name_column: Optional[str] = None
    detected_degree_column: Optional[str] = None

    column_names = list(dataframe.columns)

    for column_name in column_names:
        normalized_column = str(column_name).strip().lower()

        if detected_seat_column is None:
            exact_match_expected_seat = expected_seat_name != "" and normalized_column == expected_seat_name
            contains_expected_seat = expected_seat_name != "" and expected_seat_name in normalized_column
            contains_arabic_seat_word = "جلوس" in normalized_column or "رقم" in normalized_column
            contains_english_seat_word = "seat" in normalized_column or "seating" in normalized_column or "roll" in normalized_column

            if exact_match_expected_seat:
                detected_seat_column = column_name
                continue

            if contains_expected_seat:
                detected_seat_column = column_name
                continue

            if contains_arabic_seat_word:
                detected_seat_column = column_name
                continue

            if contains_english_seat_word:
                detected_seat_column = column_name
                continue

        if detected_name_column is None:
            exact_match_expected_name = expected_name_name != "" and normalized_column == expected_name_name
            contains_expected_name = expected_name_name != "" and expected_name_name in normalized_column
            contains_arabic_name_word = "اسم" in normalized_column
            contains_english_name_word = "name" in normalized_column or "arabic_name" in normalized_column

            if exact_match_expected_name:
                detected_name_column = column_name
                continue

            if contains_expected_name:
                detected_name_column = column_name
                continue

            if contains_arabic_name_word:
                detected_name_column = column_name
                continue

            if contains_english_name_word:
                detected_name_column = column_name
                continue

        if detected_degree_column is None:
            exact_match_expected_degree = expected_degree_name != "" and normalized_column == expected_degree_name
            contains_expected_degree = expected_degree_name != "" and expected_degree_name in normalized_column
            contains_arabic_degree_word = "درجة" in normalized_column
            contains_english_degree_word = "degree" in normalized_column or "total_degree" in normalized_column or "score" in normalized_column

            if exact_match_expected_degree:
                detected_degree_column = column_name
                continue

            if contains_expected_degree:
                detected_degree_column = column_name
                continue

            if contains_arabic_degree_word:
                detected_degree_column = column_name
                continue

            if contains_english_degree_word:
                detected_degree_column = column_name
                continue

    return {
        "seat_column": detected_seat_column,
        "name_column": detected_name_column,
        "degree_column": detected_degree_column
    }


def is_all_digits(text: str) -> bool:
    stripped_text = text.strip()
    match_result = re.fullmatch(r"\d+", stripped_text)
    is_digits = match_result is not None
    return is_digits


def try_parse_float(value: Any) -> Optional[float]:
    try:
        float_value = float(value)
        return float_value
    except Exception:
        return None


def search_dataframe_by_name(dataframe: pd.DataFrame, name_column: str, search_term: str, limit: int) -> pd.DataFrame:
    series_values = dataframe[name_column].astype(str)
    boolean_mask = series_values.str.contains(search_term, case=False, na=False)
    matched = dataframe[boolean_mask]
    limited_rows = matched.head(limit)
    return limited_rows


def search_dataframe_by_seat(dataframe: pd.DataFrame, seat_column: str, search_term: str, limit: int) -> pd.DataFrame:
    series_values = dataframe[seat_column].astype(str)
    stripped_series_values = series_values.str.strip()
    exact_mask = stripped_series_values == search_term
    matched_exact = dataframe[exact_mask]

    if len(matched_exact.index) > 0:
        limited_rows = matched_exact.head(limit)
        return limited_rows

    contains_mask = stripped_series_values.str.contains(search_term, case=False, na=False)
    matched_contains = dataframe[contains_mask]
    limited_rows = matched_contains.head(limit)
    return limited_rows


@app.on_event("startup")
def load_all_on_startup() -> None:
    for year in DB_FILES.keys():
        try:
            ensure_loaded_for_year(year)
        except Exception:
            continue


@app.post("/realode")
def reload_all() -> Dict[str, Any]:
    details: Dict[str, Any] = {}

    for year_key, config_entry in DB_FILES.items():
        csv_path = config_entry.get("file")

        file_exists_at_path = os.path.exists(csv_path)

        if not file_exists_at_path:
            details[year_key] = {"ok": False, "error": "file_not_found"}
            continue

        df_loaded = read_csv(csv_path)

        with DATA_LOCK:
            DATAFRAME_CACHE[year_key] = df_loaded

        loaded_rows = len(df_loaded.index)

        details[year_key] = {"ok": True, "loaded_rows": loaded_rows}

    response_payload = {"status": "ok", "details": details}

    return response_payload


@app.post("/{year_key}/realode")
def reload_year(year_key: str = Path(...)):
    if year_key not in DB_FILES:
        raise HTTPException(status_code=404, detail="Year not configured")

    csv_path = DB_FILES[year_key].get("file")

    file_exists_at_path = os.path.exists(csv_path)

    if not file_exists_at_path:
        raise HTTPException(status_code=404, detail="CSV file not found")

    df_loaded = read_csv(csv_path)

    with DATA_LOCK:
        DATAFRAME_CACHE[year_key] = df_loaded

    loaded_rows = len(df_loaded.index)

    response_payload = {"status": "ok", "loaded_rows": loaded_rows}

    return response_payload


@app.get("/{year_key}/search/{query}", response_model=SearchResponse)
def search_year(
    year_key: str = Path(..., description="Year key to search"),
    query: str = Path(..., description="Search term"),
    limit: int = Query(100, ge=1, le=1000)
) -> Dict[str, Any]:
    if year_key not in DB_FILES:
        raise HTTPException(status_code=404, detail="Year not configured")

    ensure_loaded_for_year(year_key)

    with DATA_LOCK:
        df_copy = DATAFRAME_CACHE[year_key].copy()

    columns_detected = detect_columns(year_key, df_copy)

    seat_column = columns_detected.get("seat_column")
    name_column = columns_detected.get("name_column")
    degree_column = columns_detected.get("degree_column")

    if seat_column is None and name_column is None:
        raise HTTPException(status_code=500, detail="CSV does not contain identifiable name or seat columns")

    normalized_query = query.strip()

    query_is_numeric = is_all_digits(normalized_query)

    if query_is_numeric:
        if seat_column is None:
            raise HTTPException(status_code=400, detail="Seat column not found in CSV")

        matched_frame = search_dataframe_by_seat(
            dataframe=df_copy,
            seat_column=seat_column,
            search_term=normalized_query,
            limit=limit
        )
    else:
        if name_column is None:
            raise HTTPException(status_code=400, detail="Name column not found in CSV")

        matched_frame = search_dataframe_by_name(
            dataframe=df_copy,
            name_column=name_column,
            search_term=normalized_query,
            limit=limit
        )

    total_matches = len(matched_frame.index)

    items_list: List[Dict[str, Any]] = []

    max_degree_value = DB_FILES[year_key].get("max_degree", None)

    for _, row in matched_frame.iterrows():
        seat_value = None
        name_value = None
        degree_value = None
        percentage_value: Optional[float] = None

        if seat_column in row.index:
            seat_value = row[seat_column]

        if name_column in row.index:
            name_value = row[name_column]

        if degree_column in row.index:
            degree_value = row[degree_column]

        parsed_degree = try_parse_float(degree_value)

        if parsed_degree is not None and max_degree_value is not None and max_degree_value != 0:
            raw_percentage = parsed_degree / float(max_degree_value)
            percentage_value = raw_percentage * 100.0

        item = {
            "seat": seat_value,
            "name": name_value,
            "degree": degree_value,
            "percentage": percentage_value
        }

        items_list.append(item)

    results_payload = {
        "total_matches": total_matches,
        "items": items_list
    }

    response_payload = {
        "results": results_payload
    }

    return response_payload
