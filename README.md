# AI Support Ticket Data Pipeline

## Project Overview
This project processes support ticket data using Python and Pandas. It loads, validates, cleans, and analyses support ticket datasets. It is the reliable data foundation (Day 1) for the AI Support Intelligence Platform.

## Project Structure
- `tickets.csv` – Original (raw) support ticket dataset. It is never modified.
- `output/cleaned_tickets.csv` – Cleaned dataset, created when you run `main.py` (not committed).
- `ticket_processor.py` – `TicketDataProcessor`: loading, validation, cleaning, custom exceptions.
- `ticket_analytics.py` – `TicketAnalytics`: ticket analytics.
- `main.py` – Runs the complete pipeline.
- `main.ipynb` – Notebook for experimentation.
- `tests/test_ticket_processor.py` – Automated tests.
- `test_results.txt` – Actual output of the last test run.

## Technologies Used
- Python
- Pandas
- NumPy
- Git and GitHub
- Pytest (for automated testing)

## Features
1. CSV data loading with clear errors (missing file, malformed CSV, empty dataset).
2. Required column validation (the error names every missing column).
3. Duplicate ticket identification and removal (rule below).
4. Missing and invalid value handling (rules below).
5. Data cleaning into a separate result; the raw CSV is preserved.
6. Ticket analytics:
   - Tickets by priority.
   - Average resolution time by category.
   - Average customer rating.
   - Unresolved ticket count.
   - Missing values and invalid records.

## Setup and How to Run

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python3 main.py
```

## Testing

```bash
python3 -m pytest -v
```

## Assumptions
- Rows are in chronological order, so for a repeated ticket ID the first row is the original entry.
- Valid ratings are whole numbers from 1 to 5; resolution time cannot be negative.
- Unresolved means status `Open`, `Pending` or `In Progress`.
- The file is a comma-separated, UTF-8 CSV.

## Data Cleaning Rules
| Problem | Rule |
|---|---|
| Extra spaces / mixed case | Text is stripped; category, priority and status are converted to Title Case |
| Blank or `NA`/`null` cells | Treated as missing |
| Missing `ticket_id` | Row removed (the ticket cannot be identified); counted in the report |
| Duplicate `ticket_id` | The **first** occurrence is kept, later ones are removed and counted |
| Non-numeric `resolution_time_hours` / `customer_rating` (e.g. `invalid`) | Becomes missing; counted |
| Negative resolution time | Treated as invalid, becomes missing; counted |
| Rating outside 1–5 or not a whole number | Treated as invalid, becomes missing; counted |
| Missing category / priority / description / status | Filled with `Unknown` |
| Missing numbers | Left missing by default so averages are not distorted. With `TicketDataProcessor(path, impute_missing=True)` they are filled with the column median |
| Data types | `resolution_time_hours` → float, `customer_rating` → nullable integer |

The original CSV is only read, never written. `save_cleaned_data` refuses to write to the raw file's path.

## Limitations
- Keeping the first duplicate is a default rule; the real business rule (e.g. keep the latest update) should be confirmed.
- The whole file is loaded into memory.
- Valid priorities/statuses are not checked against a fixed list; only spacing and case are normalised.
- Average resolution time includes every ticket that has a valid time, whatever its status.
- Median imputation (if enabled) uses the whole dataset, so it must not be used before a train/test split in ML work (data leakage).

## Project Status
Day 1 – Python Engineering and Ticket Data Pipeline.
