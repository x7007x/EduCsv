from typing import Optional, List, Dict, Any
import os
import threading

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pandas as pd

DATA_LOCK = threading.Lock()
DATA_FRAME: Optional[pd.DataFrame] = None
CSV_FILE_DEFAULT = "Database2024_Stage_New_Search.csv"
HEADERS_MAPPING = {"seat": "رقم الجلوس", "name": "الاسم", "degree": "الدرجة"}
MAX_DEGREE_VALUE = 410.0

app = FastAPI(title="Database2024 API")


class SearchResponseItem(BaseModel):
    seat: Any
    name: Any
    degree: Any


class SearchResponse(BaseModel):
    meta: Dict[str, Any]
    results: List[SearchResponseItem]


def load_csv_to_memory(csv_path: str) -> pd.DataFrame:
    dataframe = pd.read_csv(csv_path)
    return dataframe


def ensure_data_loaded() -> None:
    global DATA_FRAME
    if DATA_FRAME is None:
        csv_path = os.environ.get("DATABASE_CSV_PATH", CSV_FILE_DEFAULT)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at: {csv_path}")
        loaded_frame = load_csv_to_memory(csv_path)
        with DATA_LOCK:
            DATA_FRAME = loaded_frame


@app.on_event("startup")
def startup_load_data() -> None:
    try:
        ensure_data_loaded()
    except FileNotFoundError:
        pass


@app.post("/realode")
def reload_csv() -> Dict[str, Any]:
    global DATA_FRAME
    csv_path = os.environ.get("DATABASE_CSV_PATH", CSV_FILE_DEFAULT)
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="CSV file not found")
    new_frame = load_csv_to_memory(csv_path)
    with DATA_LOCK:
        DATA_FRAME = new_frame
    return {"status": "ok", "message": "CSV reloaded into memory", "loaded_rows": len(DATA_FRAME.index)}


@app.get("/2024/search", response_model=SearchResponse)
def search_2024(
    term: str = Query(..., description="Search term: name or seat"),
    by: Optional[str] = Query(None, description="Specify 'name' or 'seat' to limit search"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results to return")
) -> Dict[str, Any]:
    ensure_data_loaded()
    with DATA_LOCK:
        working_frame = DATA_FRAME.copy()

    column_names = list(working_frame.columns)

    name_column = None
    seat_column = None
    degree_column = None

    for column in column_names:
        lowered = column.strip().lower()
        if lowered in ("name", "full_name", "student_name"):
            name_column = column
        if lowered in ("seat", "seat_no", "seat_number", "roll"):
            seat_column = column
        if lowered in ("degree", "score", "marks"):
            degree_column = column

    if name_column is None:
        for column in column_names:
            if "name" in column.lower():
                name_column = column
                break

    if seat_column is None:
        for column in column_names:
            if "seat" in column.lower() or "roll" in column.lower():
                seat_column = column
                break

    if degree_column is None:
        for column in column_names:
            if "deg" in column.lower() or "score" in column.lower() or "mark" in column.lower():
                degree_column = column
                break

    if name_column is None and seat_column is None:
        raise HTTPException(status_code=500, detail="CSV does not contain identifiable name or seat columns")

    matches_frame = working_frame.iloc[0:0]

    search_term = term.strip()
    if by is None:
        if name_column is not None:
            name_mask = working_frame[name_column].astype(str).str.contains(search_term, case=False, na=False)
            matches_frame = pd.concat([matches_frame, working_frame[name_mask]])
        if seat_column is not None:
            seat_mask = working_frame[seat_column].astype(str).str.contains(search_term, case=False, na=False)
            matches_frame = pd.concat([matches_frame, working_frame[seat_mask]])
    else:
        search_by = by.strip().lower()
        if search_by == "name":
            if name_column is None:
                raise HTTPException(status_code=400, detail="Name column not found in CSV")
            name_mask = working_frame[name_column].astype(str).str.contains(search_term, case=False, na=False)
            matches_frame = working_frame[name_mask]
        elif search_by == "seat":
            if seat_column is None:
                raise HTTPException(status_code=400, detail="Seat column not found in CSV")
            seat_mask = working_frame[seat_column].astype(str).str.contains(search_term, case=False, na=False)
            matches_frame = working_frame[seat_mask]
        else:
            raise HTTPException(status_code=400, detail="Invalid 'by' parameter. Use 'name' or 'seat'")

    if matches_frame.empty:
        results_list = []
    else:
        limited_frame = matches_frame.head(limit)
        results_list = []
        for _, row in limited_frame.iterrows():
            seat_value = row[seat_column] if seat_column in row.index else None
            name_value = row[name_column] if name_column in row.index else None
            degree_value = row[degree_column] if degree_column in row.index else None
            result_item = {
                "seat": seat_value,
                "name": name_value,
                "degree": degree_value
            }
            results_list.append(result_item)

    response_payload = {
        "meta": {
            "headers": HEADERS_MAPPING,
            "max_degree": MAX_DEGREE_VALUE,
            "columns_detected": {
                "name_column": name_column,
                "seat_column": seat_column,
                "degree_column": degree_column
            },
            "total_matches": len(matches_frame.index)
        },
        "results": results_list
    }

    return response_payload
