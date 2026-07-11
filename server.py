import os
import re
import math
import uuid
import json
import sqlite3
import threading
import time
import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

app = FastAPI(title="OptiPrice AI Backend", version="2.1.0")

# Allow CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "./data/optiprice.db"
UPLOAD_DIR = "./data/uploads"
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ----------------------------------------------------
# DATABASE INITIALIZATION
# ----------------------------------------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    
    # Datasets metadata table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS datasets (
        id TEXT PRIMARY KEY,
        org_id TEXT NOT NULL,
        filename TEXT NOT NULL,
        row_count INTEGER DEFAULT 0,
        sku_count INTEGER DEFAULT 0,
        status TEXT NOT NULL, -- 'uploaded', 'parsing', 'validating', 'filtered', 'queued', 'analyzing', 'complete', 'failed'
        progress_pct INTEGER DEFAULT 0,
        validation_report TEXT, -- JSON string
        filters TEXT, -- JSON string
        error_message TEXT,
        created_at TEXT NOT NULL,
        analyzed_row_count INTEGER DEFAULT 0,
        analyzed_sku_count INTEGER DEFAULT 0
    )
    """)

    # Raw sales observations table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sales_observations (
        id TEXT PRIMARY KEY,
        org_id TEXT NOT NULL,
        dataset_id TEXT NOT NULL,
        sku TEXT NOT NULL,
        date TEXT NOT NULL,
        price REAL,
        units_sold REAL,
        competitor_price REAL,
        promo REAL,
        unit_cost REAL,
        category TEXT,
        is_filtered INTEGER DEFAULT 0
    )
    """)
    # Indices for performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sales_obs_lookup ON sales_observations (org_id, sku, date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sales_obs_dataset ON sales_observations (dataset_id)")

    # SKU Elasticity results table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sku_elasticity_results (
        id TEXT PRIMARY KEY,
        org_id TEXT NOT NULL,
        dataset_id TEXT NOT NULL,
        sku TEXT NOT NULL,
        elasticity_coef REAL,
        std_err REAL,
        p_value REAL,
        r_squared REAL,
        confidence_flag TEXT, -- 'high confidence', 'low confidence', 'failed'
        error_message TEXT
    )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_elasticity_results ON sku_elasticity_results (org_id, dataset_id)")

    # Support inquiries table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS support_inquiries (
        id TEXT PRIMARY KEY,
        org_id TEXT NOT NULL,
        user_id TEXT NOT NULL,
        dataset_id TEXT,
        subject TEXT NOT NULL,
        message TEXT NOT NULL,
        status TEXT NOT NULL, -- 'open', 'resolved'
        created_at TEXT NOT NULL
    )
    """)

    conn.commit()
    conn.close()

init_db()

# ----------------------------------------------------
# COLUMN ALIASES MAP
# ----------------------------------------------------
COLUMNS_MAPPING = {
    'sku': ['sku', 'product_id', 'sku_id', 'item_id', 'product'],
    'date': ['date', 'week', 'order_date', 'timestamp', 'time'],
    'price': ['price', 'price_usd', 'selling_price', 'unit_price'],
    'units_sold': ['units_sold', 'qty', 'quantity', 'sales', 'volume'],
    'competitor_price': ['competitor_price', 'competitor', 'comp_price', 'competitor_price_usd'],
    'promo': ['promo', 'promotion', 'discount_active', 'is_promo'],
    'unit_cost': ['unit_cost', 'cost', 'cog', 'cogs', 'cost_of_goods'],
    'category': ['category', 'dept', 'department', 'group', 'class']
}

def resolve_columns(df_cols: List[str]) -> Dict[str, str]:
    mapping = {}
    for col in df_cols:
        col_lower = col.lower().strip().replace(" ", "_")
        for standard, aliases in COLUMNS_MAPPING.items():
            if col_lower in aliases:
                mapping[col] = standard
                break
    return mapping

# ----------------------------------------------------
# VALIDATION RUNNER (DB-BACKED QUEUE WORKER)
# ----------------------------------------------------
def process_upload_task(dataset_id: str, file_path: str, org_id: str):
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        cursor.execute("UPDATE datasets SET status = 'parsing', progress_pct = 10 WHERE id = ?", (dataset_id,))
        conn.commit()

        # Inspect headers
        first_chunk = next(pd.read_csv(file_path, chunksize=5, keep_default_na=True))
        headers = list(first_chunk.columns)
        col_map = resolve_columns(headers)

        required = ['sku', 'date', 'price', 'units_sold']
        missing_required = [r for r in required if r not in col_map.values()]
        
        if missing_required:
            report = {
                "status": "blocking_errors",
                "row_count": 0,
                "sku_count": 0,
                "duplicate_row_count": 0,
                "missing_columns": missing_required,
                "issues": [
                    {
                        "type": "missing_required_columns",
                        "column": ", ".join(missing_required),
                        "affected_rows": 0,
                        "severity": "error",
                        "detail": f"Missing required columns: {', '.join(missing_required)}"
                    }
                ]
            }
            cursor.execute(
                "UPDATE datasets SET status = 'failed', progress_pct = 100, validation_report = ?, error_message = ? WHERE id = ?",
                (json.dumps(report), f"Missing required columns: {', '.join(missing_required)}", dataset_id)
            )
            conn.commit()
            return

        # Chunk-read and store in SQLite
        chunksize = 20000
        total_rows = 0

        cursor.execute("UPDATE datasets SET status = 'parsing', progress_pct = 30 WHERE id = ?", (dataset_id,))
        conn.commit()

        for chunk in pd.read_csv(file_path, chunksize=chunksize, keep_default_na=True):
            chunk = chunk.rename(columns=col_map)
            matched_cols = [c for c in chunk.columns if c in COLUMNS_MAPPING.keys()]
            chunk = chunk[matched_cols]

            for opt in ['competitor_price', 'promo', 'unit_cost', 'category']:
                if opt not in chunk.columns:
                    chunk[opt] = None

            chunk['id'] = [str(uuid.uuid4()) for _ in range(len(chunk))]
            chunk['org_id'] = org_id
            chunk['dataset_id'] = dataset_id
            chunk['is_filtered'] = 0

            db_cols = ['id', 'org_id', 'dataset_id', 'sku', 'date', 'price', 'units_sold', 'competitor_price', 'promo', 'unit_cost', 'category', 'is_filtered']
            chunk = chunk[db_cols]
            
            chunk.to_sql('sales_observations', conn, if_exists='append', index=False)
            total_rows += len(chunk)

        cursor.execute("UPDATE datasets SET status = 'validating', progress_pct = 60 WHERE id = ?", (dataset_id,))
        conn.commit()

        # RUN VALIDATION ON DATABASE
        df = pd.read_sql_query(
            "SELECT sku, date, price, units_sold, competitor_price, promo, unit_cost, category FROM sales_observations WHERE dataset_id = ?",
            conn, params=(dataset_id,)
        )
        
        sku_count = df['sku'].nunique()
        missing_cols_report = [c for c in COLUMNS_MAPPING.keys() if c not in col_map.values()]

        issues = []
        blocking = False

        # Clean/coerce numbers
        df['price_num'] = pd.to_numeric(df['price'], errors='coerce')
        df['units_sold_num'] = pd.to_numeric(df['units_sold'], errors='coerce')
        parsed_dates = pd.to_datetime(df['date'], errors='coerce')

        # 1. Null values or type mismatch in required columns
        null_mask = df['sku'].isna() | (df['sku'] == '') | df['date'].isna() | df['price'].isna() | df['units_sold'].isna()
        type_mask = (df['price'].notna() & df['price_num'].isna()) | (df['units_sold'].notna() & df['units_sold_num'].isna())
        missing_count = (null_mask | type_mask).sum()
        if missing_count > 0:
            issues.append({
                "type": "missing_required_data",
                "affected_rows": int(missing_count),
                "severity": "error",
                "detail": f"MISSING REQUIRED DATA: {missing_count} rows are missing a required field or contain non-numeric price/units."
            })
            blocking = True

        # 2. Price <= 0
        neg_prices = (df['price_num'] <= 0).sum()
        if neg_prices > 0:
            issues.append({
                "type": "invalid_price",
                "column": "price",
                "affected_rows": int(neg_prices),
                "severity": "error",
                "detail": f"INVALID PRICE: {neg_prices} rows have price <= 0. These rows must be excluded before analysis."
            })
            blocking = True

        # 3. Units sold < 0 (note: units_sold = 0 is fine, but < 0 is blocking)
        neg_units = (df['units_sold_num'] < 0).sum()
        if neg_units > 0:
            issues.append({
                "type": "invalid_units",
                "column": "units_sold",
                "affected_rows": int(neg_units),
                "severity": "error",
                "detail": f"INVALID UNITS: {neg_units} rows have negative units_sold. These rows must be excluded before analysis."
            })
            blocking = True

        # 4. Date parsing check
        invalid_dates = parsed_dates.isna().sum()
        if invalid_dates > 0:
            issues.append({
                "type": "invalid_date",
                "affected_rows": int(invalid_dates),
                "severity": "error",
                "detail": f"INVALID DATE: {invalid_dates} rows have an unparseable date."
            })
            blocking = True

        # 5. Duplicate SKU + Date check
        duplicates = df.duplicated(subset=['sku', 'date']).sum()
        if duplicates > 0:
            issues.append({
                "type": "duplicate_rows",
                "affected_rows": int(duplicates),
                "severity": "warning",
                "detail": f"DUPLICATE OBSERVATIONS: {duplicates} duplicate SKU + Date rows found. Move to Filter step to configure handling."
            })

        # 6. Insufficient history or price variation per SKU
        sku_groups = df.groupby('sku')
        insufficient_history_skus = []
        no_price_variation_skus = []

        for sku_val, group in sku_groups:
            valid_len = len(group[(group['price_num'] > 0) & (group['units_sold_num'] >= 0)])
            if valid_len < 10:
                insufficient_history_skus.append(sku_val)
            else:
                unique_prices = group['price_num'].nunique()
                if unique_prices <= 1:
                    no_price_variation_skus.append(sku_val)

        if insufficient_history_skus:
            issues.append({
                "type": "insufficient_history",
                "skus": insufficient_history_skus[:100],
                "severity": "warning",
                "detail": f"Fewer than 10 observations — elasticity will not be estimated for these {len(insufficient_history_skus)} SKUs."
            })

        if no_price_variation_skus:
            issues.append({
                "type": "no_price_variation",
                "skus": no_price_variation_skus[:100],
                "severity": "warning",
                "detail": f"Zero price variation — elasticity cannot be estimated for these {len(no_price_variation_skus)} SKUs."
            })

        # 7. Optional missing reports
        for opt in ['competitor_price', 'promo', 'unit_cost']:
            if opt in col_map.values():
                missing_opt = df[opt].isna().sum()
                if missing_opt > 0:
                    issues.append({
                        "type": "missing_values",
                        "column": opt,
                        "affected_rows": int(missing_opt),
                        "severity": "warning",
                        "detail": f"{missing_opt} rows are missing optional field: '{opt}'"
                    })

        # Sort issues so errors render above warnings
        issues.sort(key=lambda x: 0 if x['severity'] == 'error' else 1)

        status_str = "issues_found"
        if blocking:
            status_str = "issues_found" # Keep as issues_found so they can proceed to filter step!
        elif not issues:
            status_str = "clean"

        report = {
            "status": "blocking_errors" if blocking else status_str,
            "row_count": total_rows,
            "sku_count": sku_count,
            "duplicate_row_count": int(duplicates),
            "missing_columns": missing_cols_report,
            "issues": issues
        }

        # Update dataset record
        cursor.execute(
            "UPDATE datasets SET status = ?, progress_pct = 100, row_count = ?, sku_count = ?, validation_report = ? WHERE id = ?",
            ("filtered", total_rows, sku_count, json.dumps(report), dataset_id)
        )
        conn.commit()

    except Exception as e:
        import traceback
        err_msg = f"{str(e)}\n{traceback.format_exc()}"
        cursor.execute("UPDATE datasets SET status = 'failed', error_message = ?, progress_pct = 100 WHERE id = ?", (err_msg, dataset_id))
        conn.commit()
    finally:
        conn.close()

# ----------------------------------------------------
# ELASTICITY ENGINE REGRESSION TASK (DB-BACKED)
# ----------------------------------------------------
def run_analysis_task(dataset_id: str, filters: Dict[str, Any], org_id: str):
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        cursor.execute("UPDATE datasets SET status = 'analyzing', progress_pct = 5 WHERE id = ?", (dataset_id,))
        conn.commit()

        # Get raw data
        df = pd.read_sql_query(
            "SELECT * FROM sales_observations WHERE dataset_id = ?",
            conn, params=(dataset_id,)
        )

        if df.empty:
            raise ValueError("No observations found in dataset")

        # Parse date
        df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date_parsed'])

        # Apply date filters
        if filters.get('start_date'):
            df = df[df['date_parsed'] >= pd.to_datetime(filters['start_date'])]
        if filters.get('end_date'):
            df = df[df['date_parsed'] <= pd.to_datetime(filters['end_date'])]

        # Category filter
        if filters.get('categories') and isinstance(filters['categories'], list) and len(filters['categories']) > 0:
            df = df[df['category'].isin(filters['categories'])]

        # Exclude SKUs with insufficient history
        if filters.get('exclude_insufficient', True):
            counts = df.groupby('sku').size()
            good_skus = counts[counts >= 10].index
            df = df[df['sku'].isin(good_skus)]

        # Drop rows where required/clean fields are null/negative/zero
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['units_sold'] = pd.to_numeric(df['units_sold'], errors='coerce')
        df = df.dropna(subset=['price', 'units_sold'])
        
        # Exclude rows with missing optional fields
        if filters.get('exclude_missing_optional', False):
            opt_cols = []
            if 'competitor_price' in df.columns and df['competitor_price'].notna().sum() > 0:
                opt_cols.append('competitor_price')
            if 'promo' in df.columns and df['promo'].notna().sum() > 0:
                opt_cols.append('promo')
            if opt_cols:
                df = df.dropna(subset=opt_cols)

        # DUPLICATE-ROW RESOLUTION STEP (Explicit choice)
        duplicate_handling = filters.get('duplicate_handling', 'average_sum')
        if duplicate_handling == 'average_sum':
            agg_dict = {
                'price': 'mean',
                'units_sold': 'sum'
            }
            if 'competitor_price' in df.columns and df['competitor_price'].notna().sum() > 0:
                agg_dict['competitor_price'] = 'mean'
            if 'promo' in df.columns and df['promo'].notna().sum() > 0:
                agg_dict['promo'] = 'max'
            if 'unit_cost' in df.columns and df['unit_cost'].notna().sum() > 0:
                agg_dict['unit_cost'] = 'mean'
            if 'category' in df.columns and df['category'].notna().sum() > 0:
                agg_dict['category'] = 'first'
                
            df = df.groupby(['sku', 'date_parsed'], as_index=False).agg(agg_dict)
            df['date'] = df['date_parsed'].dt.strftime('%Y-%m-%d')
        elif duplicate_handling == 'keep_first':
            df = df.drop_duplicates(subset=['sku', 'date_parsed'], keep='first')
        elif duplicate_handling == 'keep_last':
            df = df.drop_duplicates(subset=['sku', 'date_parsed'], keep='last')

        # Update analyzed count in dataset
        analyzed_row_count = len(df)
        analyzed_sku_count = df['sku'].nunique()
        cursor.execute(
            "UPDATE datasets SET analyzed_row_count = ?, analyzed_sku_count = ? WHERE id = ?",
            (analyzed_row_count, analyzed_sku_count, dataset_id)
        )
        conn.commit()

        if analyzed_sku_count == 0:
            cursor.execute(
                "UPDATE datasets SET status = 'failed', progress_pct = 100, error_message = 'No SKUs left to analyze after filtering' WHERE id = ?",
                (dataset_id,)
            )
            conn.commit()
            return

        skus = df['sku'].unique()
        total_skus = len(skus)
        completed = 0

        # Remove existing results
        cursor.execute("DELETE FROM sku_elasticity_results WHERE dataset_id = ?", (dataset_id,))
        conn.commit()

        for sku in skus:
            sku_df = df[df['sku'] == sku].copy()
            
            if sku_df['price'].nunique() <= 1:
                cursor.execute(
                    "INSERT INTO sku_elasticity_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (str(uuid.uuid4()), org_id, dataset_id, sku, None, None, None, None, 'failed', 'No price variation')
                )
                completed += 1
                progress = int((completed / total_skus) * 90) + 5
                cursor.execute("UPDATE datasets SET progress_pct = ? WHERE id = ?", (progress, dataset_id))
                conn.commit()
                continue
                
            try:
                sku_df['log_units'] = np.log(sku_df['units_sold'])
                sku_df['log_price'] = np.log(sku_df['price'])
                
                # Seasonality terms
                sku_df['week_of_year'] = sku_df['date_parsed'].dt.isocalendar().week
                sku_df['sin_week'] = np.sin(2 * np.pi * sku_df['week_of_year'] / 52.0)
                sku_df['cos_week'] = np.cos(2 * np.pi * sku_df['week_of_year'] / 52.0)

                X_cols = ['log_price']
                
                # competitor price
                if 'competitor_price' in sku_df.columns:
                    if not filters.get('exclude_missing_optional', False):
                        median_comp = sku_df['competitor_price'].median()
                        if pd.isna(median_comp) or median_comp <= 0:
                            median_comp = sku_df['price'].median()
                        sku_df['competitor_price'] = sku_df['competitor_price'].fillna(median_comp)
                    
                    sku_df['competitor_price'] = pd.to_numeric(sku_df['competitor_price'], errors='coerce')
                    sku_df = sku_df[sku_df['competitor_price'] > 0]
                    
                    if len(sku_df) >= 10 and sku_df['competitor_price'].nunique() > 1:
                        sku_df['log_competitor_price'] = np.log(sku_df['competitor_price'])
                        X_cols.append('log_competitor_price')

                # promo
                if 'promo' in sku_df.columns:
                    if not filters.get('exclude_missing_optional', False):
                        sku_df['promo'] = sku_df['promo'].fillna(0)
                    
                    sku_df['promo'] = pd.to_numeric(sku_df['promo'], errors='coerce').fillna(0)
                    if len(sku_df) >= 10 and sku_df['promo'].nunique() > 1:
                        X_cols.append('promo')

                # seasonality
                if len(sku_df) >= 20:
                    X_cols.extend(['sin_week', 'cos_week'])

                y = sku_df['log_units']
                X = sku_df[X_cols]
                X = sm.add_constant(X)

                model = sm.OLS(y, X)
                results = model.fit()

                elasticity_coef = float(results.params['log_price'])
                std_err = float(results.bse['log_price'])
                p_value = float(results.pvalues['log_price'])
                r_squared = float(results.rsquared)

                if p_value > 0.05 or r_squared < 0.3 or math.isnan(elasticity_coef):
                    conf = 'low confidence'
                else:
                    conf = 'high confidence'

                cursor.execute(
                    "INSERT INTO sku_elasticity_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (str(uuid.uuid4()), org_id, dataset_id, sku, elasticity_coef, std_err, p_value, r_squared, conf, None)
                )

            except Exception as inner_e:
                cursor.execute(
                    "INSERT INTO sku_elasticity_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (str(uuid.uuid4()), org_id, dataset_id, sku, None, None, None, None, 'failed', str(inner_e))
                )

            completed += 1
            progress = int((completed / total_skus) * 90) + 5
            cursor.execute("UPDATE datasets SET progress_pct = ? WHERE id = ?", (progress, dataset_id))
            conn.commit()

        cursor.execute("UPDATE datasets SET status = 'complete', progress_pct = 100 WHERE id = ?", (dataset_id,))
        conn.commit()

    except Exception as e:
        import traceback
        err_msg = f"{str(e)}\n{traceback.format_exc()}"
        cursor.execute("UPDATE datasets SET status = 'failed', error_message = ?, progress_pct = 100 WHERE id = ?", (err_msg, dataset_id))
        conn.commit()
    finally:
        conn.close()

# ----------------------------------------------------
# LIGHTWEIGHT DB-BACKED POLLING QUEUE WORKER
# ----------------------------------------------------
def db_worker_loop():
    print("[WORKER] SQLite DB-backed job worker thread initialized.")
    while True:
        try:
            conn = get_db()
            cursor = conn.cursor()
            
            # 1. Check for newly uploaded files needing validation
            cursor.execute("SELECT id, filename, org_id FROM datasets WHERE status = 'uploaded' LIMIT 1")
            row = cursor.fetchone()
            if row:
                dataset_id, filename, org_id = row['id'], row['filename'], row['org_id']
                file_path = os.path.join(UPLOAD_DIR, f"{dataset_id}.csv")
                print(f"[WORKER] Found uploaded dataset: {filename} ({dataset_id}). Starting validation...")
                conn.close()
                process_upload_task(dataset_id, file_path, org_id)
                continue

            # 2. Check for queued estimation runs
            cursor.execute("SELECT id, filters, org_id FROM datasets WHERE status = 'queued' LIMIT 1")
            row = cursor.fetchone()
            if row:
                dataset_id, filters_str, org_id = row['id'], row['filters'], row['org_id']
                filters = json.loads(filters_str) if filters_str else {}
                print(f"[WORKER] Found queued regression request for dataset {dataset_id}. Running OLS OLS...")
                conn.close()
                run_analysis_task(dataset_id, filters, org_id)
                continue
                
            conn.close()
        except Exception as e:
            print("[WORKER] Error in database worker loop:", e)
            
        time.sleep(2)

# Start background thread
threading.Thread(target=db_worker_loop, daemon=True).start()

# ----------------------------------------------------
# API ENDPOINTS
# ----------------------------------------------------
@app.get("/api/health")
def health_check():
    return {"status": "ok", "engine": "active"}

@app.post("/api/upload")
def upload_dataset(
    file: UploadFile = File(...),
    org_id: str = Form("org_default")
):
    dataset_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    
    if file_extension.lower() != '.csv':
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    dest_path = os.path.join(UPLOAD_DIR, f"{dataset_id}.csv")
    
    with open(dest_path, "wb") as buffer:
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk:
                break
            buffer.write(chunk)

    # Insert dataset record as 'uploaded'
    conn = get_db()
    cursor = conn.cursor()
    now_str = datetime.utcnow().isoformat()
    
    cursor.execute(
        "INSERT INTO datasets (id, org_id, filename, status, progress_pct, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (dataset_id, org_id, file.filename, "uploaded", 0, now_str)
    )
    conn.commit()
    conn.close()

    return {
        "dataset_id": dataset_id,
        "filename": file.filename,
        "status": "uploaded"
    }

@app.get("/api/datasets/{dataset_id}")
def get_dataset(dataset_id: str):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Dataset not found")

    report = None
    if row['validation_report']:
        report = json.loads(row['validation_report'])

    return {
        "id": row['id'],
        "org_id": row['org_id'],
        "filename": row['filename'],
        "row_count": row['row_count'],
        "sku_count": row['sku_count'],
        "status": row['status'],
        "progress_pct": row['progress_pct'],
        "validation_report": report,
        "error_message": row['error_message'],
        "created_at": row['created_at'],
        "analyzed_row_count": row['analyzed_row_count'],
        "analyzed_sku_count": row['analyzed_sku_count']
    }

@app.get("/api/datasets/{dataset_id}/preview")
def get_dataset_preview(dataset_id: str, limit: int = 50):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT filename, validation_report FROM datasets WHERE id = ?", (dataset_id,))
    ds = cursor.fetchone()
    if not ds:
        conn.close()
        raise HTTPException(status_code=404, detail="Dataset not found")

    report = json.loads(ds['validation_report']) if ds['validation_report'] else {}
    dup_count = report.get('duplicate_row_count', 0)

    df_rows = pd.read_sql_query(
        "SELECT sku, date, price, units_sold, competitor_price, promo, unit_cost, category FROM sales_observations WHERE dataset_id = ? LIMIT ?",
        conn, params=(dataset_id, limit)
    )

    categories = pd.read_sql_query(
        "SELECT DISTINCT category FROM sales_observations WHERE dataset_id = ? AND category IS NOT NULL",
        conn, params=(dataset_id,)
    )['category'].tolist()

    dates = pd.read_sql_query(
        "SELECT MIN(date) as min_date, MAX(date) as max_date FROM sales_observations WHERE dataset_id = ?",
        conn, params=(dataset_id,)
    ).iloc[0].to_dict()

    conn.close()

    return {
        "filename": ds['filename'],
        "rows": df_rows.to_dict(orient='records'),
        "categories": categories,
        "min_date": dates.get('min_date'),
        "max_date": dates.get('max_date'),
        "duplicate_row_count": dup_count
    }

class FilterPreviewSchema(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    categories: Optional[List[str]] = None
    exclude_insufficient: bool = True
    exclude_missing_optional: bool = False
    duplicate_handling: str = "average_sum"

def apply_filters_to_df(df: pd.DataFrame, payload: FilterPreviewSchema) -> pd.DataFrame:
    if df.empty:
        return df

    df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date_parsed'])

    if payload.start_date:
        df = df[df['date_parsed'] >= pd.to_datetime(payload.start_date)]
    if payload.end_date:
        df = df[df['date_parsed'] <= pd.to_datetime(payload.end_date)]

    if payload.categories:
        df = df[df['category'].isin(payload.categories)]

    if payload.exclude_insufficient:
        counts = df.groupby('sku').size()
        good_skus = counts[counts >= 10].index
        df = df[df['sku'].isin(good_skus)]

    if payload.exclude_missing_optional:
        opt_cols = []
        if 'competitor_price' in df.columns and df['competitor_price'].notna().sum() > 0:
            opt_cols.append('competitor_price')
        if 'promo' in df.columns and df['promo'].notna().sum() > 0:
            opt_cols.append('promo')
        if opt_cols:
            df = df.dropna(subset=opt_cols)

    return df

@app.post("/api/datasets/{dataset_id}/preview-stats")
def post_dataset_preview_stats(dataset_id: str, payload: FilterPreviewSchema):
    conn = get_db()
    df = pd.read_sql_query(
        "SELECT sku, date, price, units_sold, competitor_price, promo, unit_cost, category FROM sales_observations WHERE dataset_id = ?",
        conn, params=(dataset_id,)
    )
    conn.close()

    df_filtered = apply_filters_to_df(df, payload)
    return {
        "row_count": len(df_filtered),
        "sku_count": df_filtered['sku'].nunique()
    }

@app.post("/api/datasets/{dataset_id}/analyze")
def run_analysis(
    dataset_id: str,
    payload: FilterPreviewSchema,
    org_id: str = "org_default"
):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT status, filename FROM datasets WHERE id = ?", (dataset_id,))
    row = cursor.fetchone()

    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Fetch rows to run server-side enforcement of blocking errors
    df = pd.read_sql_query(
        "SELECT sku, date, price, units_sold, competitor_price, promo, unit_cost, category FROM sales_observations WHERE dataset_id = ?",
        conn, params=(dataset_id,)
    )

    df_filtered = apply_filters_to_df(df, payload)

    # 1. Enforce price <= 0 check
    df_filtered['price_num'] = pd.to_numeric(df_filtered['price'], errors='coerce')
    neg_prices = (df_filtered['price_num'] <= 0).sum()
    if neg_prices > 0:
        conn.close()
        raise HTTPException(
            status_code=400,
            detail=f"Cannot run analysis: The active filtered dataset still contains {neg_prices} rows with price <= 0. Please filter them out or re-upload."
        )

    # 2. Enforce units_sold < 0 check
    df_filtered['units_sold_num'] = pd.to_numeric(df_filtered['units_sold'], errors='coerce')
    neg_units = (df_filtered['units_sold_num'] < 0).sum()
    if neg_units > 0:
        conn.close()
        raise HTTPException(
            status_code=400,
            detail=f"Cannot run analysis: The active filtered dataset still contains {neg_units} rows with negative units_sold. Please filter them out or re-upload."
        )

    # 3. Enforce missing required fields
    null_mask = df_filtered['sku'].isna() | (df_filtered['sku'] == '') | df_filtered['date'].isna() | df_filtered['price'].isna() | df_filtered['units_sold'].isna()
    type_mask = (df_filtered['price'].notna() & df_filtered['price_num'].isna()) | (df_filtered['units_sold'].notna() & df_filtered['units_sold_num'].isna())
    missing_count = (null_mask | type_mask).sum()
    if missing_count > 0:
        conn.close()
        raise HTTPException(
            status_code=400,
            detail=f"Cannot run analysis: The active filtered dataset still contains {missing_count} rows with missing or non-numeric required fields. Please exclude them."
        )

    # 4. Enforce date parsing check
    parsed_dates = pd.to_datetime(df_filtered['date'], errors='coerce')
    invalid_dates = parsed_dates.isna().sum()
    if invalid_dates > 0:
        conn.close()
        raise HTTPException(
            status_code=400,
            detail=f"Cannot run analysis: The active filtered dataset still contains {invalid_dates} rows with unparseable dates. Please exclude them."
        )

    # Set status to queued for DB-backed queue processing
    cursor.execute(
        "UPDATE datasets SET status = 'queued', progress_pct = 0, filters = ? WHERE id = ?",
        (json.dumps(payload.dict()), dataset_id)
    )
    conn.commit()
    conn.close()

    return {
        "dataset_id": dataset_id,
        "status": "queued"
    }

@app.get("/api/datasets/{dataset_id}/results")
def get_analysis_results(dataset_id: str):
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute("SELECT filename, analyzed_row_count, analyzed_sku_count, status FROM datasets WHERE id = ?", (dataset_id,))
    ds = cursor.fetchone()
    if not ds:
        conn.close()
        raise HTTPException(status_code=404, detail="Dataset not found")

    results_df = pd.read_sql_query(
        "SELECT sku, elasticity_coef, std_err, p_value, r_squared, confidence_flag, error_message FROM sku_elasticity_results WHERE dataset_id = ?",
        conn, params=(dataset_id,)
    )
    conn.close()

    results = results_df.to_dict(orient='records')

    high_conf_df = results_df[results_df['confidence_flag'] == 'high confidence']
    avg_elasticity = float(high_conf_df['elasticity_coef'].mean()) if not high_conf_df.empty else None
    
    return {
        "dataset_id": dataset_id,
        "filename": ds['filename'],
        "status": ds['status'],
        "analyzed_row_count": ds['analyzed_row_count'],
        "analyzed_sku_count": ds['analyzed_sku_count'],
        "avg_elasticity": avg_elasticity,
        "results": results
    }

class SupportInquiryPayload(BaseModel):
    org_id: str = "org_default"
    user_id: str = "user_default"
    dataset_id: Optional[str] = None
    subject: str
    message: str

@app.post("/api/support-inquiries")
def create_support_inquiry(payload: SupportInquiryPayload):
    conn = get_db()
    cursor = conn.cursor()
    inquiry_id = str(uuid.uuid4())
    now_str = datetime.utcnow().isoformat()
    
    cursor.execute(
        "INSERT INTO support_inquiries (id, org_id, user_id, dataset_id, subject, message, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (inquiry_id, payload.org_id, payload.user_id, payload.dataset_id, payload.subject, payload.message, "open", now_str)
    )
    conn.commit()
    conn.close()

    return {
        "inquiry_id": inquiry_id,
        "status": "success",
        "detail": "Support ticket created successfully"
    }

# Serving index.html on root
@app.get("/")
def get_index():
    return FileResponse("./index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
