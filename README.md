# OptiPrice AI: Enterprise Price Optimization & Causal Demand Simulation

OptiPrice AI is an enterprise-grade price optimization, causal elasticity modeling, and demand simulation engine. It combines log-log Ordinary Least Squares (OLS) multi-variable regression models with interactive simulation dashboards to help businesses measure price elasticity, forecast demand, and maximize operating margins in real-time.

---

## Key Modules & Capabilities

### 1. Data Intake & Validation Layer
- **High-Throughput Uploads**: Stream/chunk-reads large CSV datasets (supporting up to 50MB / ~500k rows) in Python using Pandas to prevent request timeouts on serverless or free-tier hosting.
- **Fuzzy Schema Mapping**: Automatically maps headers case-insensitively using standard aliases (e.g., `product_id`/`sku_id` $\rightarrow$ `sku`; `qty`/`quantity`/`volume` $\rightarrow$ `units_sold`).
- **Validation Engine**:
  - **Blocking Errors**: Catches invalid prices ($\le 0$), negative units sold ($< 0$), missing required fields (nulls or non-numeric types), and unparseable dates. Enforced synchronously prior to analysis.
  - **Acoustic Warnings**: Identifies thin SKU history ($< 10$ records), zero price variation, and missing optional fields.
- **Duplicate-Row Resolution**: Gives users explicit control in the UI to handle duplicate SKU + Date entries:
  - **Option A (Default)**: Average price and sum units sold.
  - **Option B**: Keep only the first occurrence.
  - **Option C**: Keep only the last occurrence.

### 2. Elasticity Estimation Engine
- **Log-Log OLS Regression**: Fits a log-log linear demand curve of the form:
  $$\ln(\text{Units Sold}) = \beta_0 + \beta_1 \ln(\text{Price}) + \beta_2 \ln(\text{Competitor Price}) + \beta_3 \text{Promo} + \beta_4 \sin\left(\frac{2\pi w}{52}\right) + \beta_5 \cos\left(\frac{2\pi w}{52}\right) + \epsilon$$
  where $\beta_1$ directly represents the **Price Elasticity of Demand (PED)** coefficient.
- **Confounder Control**: Automatically incorporates competitor prices, promotions, and double-term Fourier seasonality variables (when data volume is sufficient) to isolate true price elasticity from exogenous variables.
- **Statistical Confidence Profiling**: Computes coefficient standard errors, R-squared values, and p-values to label elasticity models with confidence flags (`high confidence` vs. `low confidence` vs. `failed`).

### 3. Simulation & Optimization Dashboard
- **Dynamic Demand Curve Visualizations**: Renders simulated demand curves dynamically based on OLS outputs.
- **Price Optimizer**: Simulates price changes ($-50\%$ to $+50\%$) to find the revenue-maximizing and profit-maximizing price points.
- **Interactive Scenarios**: Allows users to alter costs, competitive prices, and promotion schedules to preview margin outcomes instantly.

---

## Architecture & Production Deployment Stack

This codebase is designed specifically to run reliably on a **free-tier production stack**:

- **Frontend**: Next.js / HTML5 (Deployable to Vercel Free Tier).
- **Backend**: Python / FastAPI (Deployable to Render Free Web Service).
- **Background Worker**: Database-backed task table and polling daemon thread inside the FastAPI process. This queue-less setup eliminates the need for external message brokers (Redis/Celery) while ensuring task persistence across free-tier container restarts.
- **Cold-Start Handling**: The UI monitors backend connectivity via `/api/health` on load and alerts the user if the server is waking up after inactivity.
- **Database**: SQLite (local dev) or Postgres (compatible with Neon/Supabase free tiers). Includes storage ceiling warnings in the UI.

---

## File Structure

```
├── index.html               # Frontend dashboard UI, styles, and controller JS
├── server.py                 # FastAPI application, OLS engine, and polling worker
├── generate_test_data.py     # Script to generate mock clean and invalid CSV datasets
├── .gitignore                # Rules for files to exclude from Git tracking
└── test_data/                # Output folder for generated test CSV files
```

---

## Getting Started

### 1. Installation
Clone the repository and install dependencies:
```bash
pip install fastapi uvicorn pandas numpy statsmodels pydantic
```

### 2. Generating Test Datasets
To generate mock files for verifying validation and regression tasks:
```bash
python generate_test_data.py
```
This writes the following files to the `./test_data/` directory:
- `clean_dataset.csv`: Standard records with expected warnings (insufficient data, zero price variance).
- `invalid_values_dataset.csv`: Records with negative prices and non-numeric units.
- `missing_columns_dataset.csv`: Missing standard required column headers.

### 3. Run the Development Server
Launch the FastAPI server:
```bash
python server.py
```
Open [http://localhost:8000](http://localhost:8000) in your web browser.

---

## License
MIT License
