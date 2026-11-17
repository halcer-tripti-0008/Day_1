from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Meaningful exceptions. They inherit from ValueError so older code that does
# `except ValueError` still works, but callers can now catch the exact problem.
# ---------------------------------------------------------------------------
class TicketDataError(ValueError):
    """Base class for all ticket-data problems."""


class InvalidCSVError(TicketDataError):
    """The file is not a well-formed CSV."""


class EmptyDatasetError(TicketDataError):
    """The file contains no data rows."""


class MissingColumnsError(TicketDataError):
    """One or more required columns are absent."""

    def __init__(self, missing_columns):
        self.missing_columns = list(missing_columns)
        super().__init__(f"Missing required columns: {self.missing_columns}")


class TicketDataProcessor:
    """Loads, validates and cleans a support-ticket CSV.

    The raw CSV is only ever read. The cleaned data is a separate DataFrame.

    Cleaning rules (also listed in README.md):
      1. Text is stripped; category/priority/status are converted to Title Case.
      2. Rows without a ticket_id are removed.
      3. Duplicate ticket_ids: the FIRST occurrence is kept, later ones removed.
      4. resolution_time_hours / customer_rating are converted to numbers;
         non-numeric text becomes missing.
      5. Negative resolution times, ratings outside 1-5 and non-whole ratings
         are treated as invalid and become missing.
      6. Missing category/priority/description/status become "Unknown".
      7. Missing numbers stay missing, unless impute_missing=True, in which
         case they are filled with the column median.
    """

    REQUIRED_COLUMNS = [
        "ticket_id",
        "category",
        "priority",
        "description",
        "resolution_time_hours",
        "customer_rating",
        "status"
    ]

    TITLE_CASE_COLUMNS = ["category", "priority", "status"]
    TEXT_COLUMNS = ["category", "priority", "description", "status"]
    NUMERIC_COLUMNS = ["resolution_time_hours", "customer_rating"]
    RATING_MIN = 1
    RATING_MAX = 5
    UNKNOWN = "Unknown"

    def __init__(self, file_path, impute_missing=False):
        self.file_path = file_path
        self.impute_missing = impute_missing
        self.df = None
        self.cleaned_df = None
        self.duplicate_count = 0
        self.missing_report = {}
        self.invalid_report = {}

    # ------------------------------------------------------------------ load
    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"File not found: {self.file_path}"
            ) from exc
        except pd.errors.EmptyDataError as exc:
            raise EmptyDatasetError("The CSV file is empty.") from exc
        except (pd.errors.ParserError, UnicodeDecodeError) as exc:
            raise InvalidCSVError("Invalid CSV format.") from exc

        # If every row has MORE fields than the header, pandas does not fail:
        # it silently turns the extra leading fields into an index.
        if not isinstance(self.df.index, pd.RangeIndex):
            raise InvalidCSVError(
                "Invalid CSV format: rows have more fields than the header."
            )

        if self.df.empty:
            raise EmptyDatasetError("The CSV file has a header but no data rows.")

        self.df.columns = [str(col).strip() for col in self.df.columns]
        print("CSV loaded successfully!")
        return self.df

    # -------------------------------------------------------------- validate
    def validate_columns(self):
        if self.df is None:
            raise ValueError("Please load the CSV first.")

        missing_columns = [
            col for col in self.REQUIRED_COLUMNS
            if col not in self.df.columns
        ]

        if missing_columns:
            raise MissingColumnsError(missing_columns)

        print("Column validation successful!")
        return True

    # ------------------------------------------------------------ duplicates
    def handle_duplicates(self):
        """Keep the first occurrence of each ticket_id (rule 3)."""
        if self.cleaned_df is None:
            raise ValueError("Please start cleaning first (call clean_data).")

        self.duplicate_count = int(
            self.cleaned_df["ticket_id"].duplicated().sum()
        )

        self.cleaned_df = self.cleaned_df.drop_duplicates(
            subset=["ticket_id"],
            keep="first"
        ).copy()

        return self.cleaned_df

    # ----------------------------------------------------------------- clean
    def clean_data(self):
        self.validate_columns()

        self.invalid_report = {}  # a new run must not add to old counts
        self.cleaned_df = self.df.copy()  # the raw frame is never modified

        self._normalise_text()

        # Counted after stripping, so blank/space-only cells count as missing.
        self.missing_report = {
            col: int(count)
            for col, count in self.cleaned_df.isna().sum().items()
        }

        self._remove_rows_without_id()
        self.handle_duplicates()
        self._convert_numeric_columns()
        self._fill_missing_values()

        print("Data cleaning completed successfully!")
        return self.cleaned_df

    def _normalise_text(self):
        """Strip spaces, turn blank cells into missing, apply Title Case."""
        for col in self.TEXT_COLUMNS + ["ticket_id"]:
            if pd.api.types.is_numeric_dtype(self.cleaned_df[col]):
                continue  # numeric ids need no text clean-up

            text = self.cleaned_df[col].astype("string").str.strip()
            if col in self.TITLE_CASE_COLUMNS:
                text = text.str.title()
            self.cleaned_df[col] = text.replace("", pd.NA).astype(object)

    def _remove_rows_without_id(self):
        missing_ids = int(self.cleaned_df["ticket_id"].isna().sum())

        self.cleaned_df = self.cleaned_df.dropna(
            subset=["ticket_id"]
        ).copy()

        self.invalid_report["missing_ticket_ids_removed"] = missing_ids

    def _convert_numeric_columns(self):
        for col in self.NUMERIC_COLUMNS:
            original = self.cleaned_df[col]

            converted = pd.to_numeric(
                original,
                errors="coerce"
            )

            invalid_mask = original.notna() & converted.isna()

            self.invalid_report[f"invalid_{col}"] = int(
                invalid_mask.sum()
            )

            self.cleaned_df[col] = converted.astype("float64")

        negative_time = (
            self.cleaned_df["resolution_time_hours"] < 0
        )

        self.invalid_report["negative_resolution_time"] = int(
            negative_time.sum()
        )

        self.cleaned_df.loc[
            negative_time, "resolution_time_hours"
        ] = float("nan")

        # Only look at ratings that EXIST: a missing rating is "missing",
        # not "invalid" (NaN fails every comparison, so ~between() is True).
        rating = self.cleaned_df["customer_rating"]
        invalid_rating = rating.notna() & (
            ~rating.between(self.RATING_MIN, self.RATING_MAX)
            | (rating % 1 != 0)
        )

        self.invalid_report["invalid_customer_rating"] = int(
            invalid_rating.sum()
        )

        self.cleaned_df.loc[
            invalid_rating, "customer_rating"
        ] = float("nan")

    def _fill_missing_values(self):
        for col in self.TEXT_COLUMNS:
            self.cleaned_df[col] = (
                self.cleaned_df[col].fillna(self.UNKNOWN)
            )

        if self.impute_missing:
            for col in self.NUMERIC_COLUMNS:
                median_value = self.cleaned_df[col].median()

                if pd.isna(median_value):
                    median_value = (
                        0.0 if col == "resolution_time_hours" else 3.0
                    )

                if col == "customer_rating":
                    median_value = round(median_value)  # keep ratings whole

                self.cleaned_df[col] = (
                    self.cleaned_df[col].fillna(median_value)
                )

        # Ratings are whole numbers; "Int64" can also hold missing values.
        self.cleaned_df["customer_rating"] = (
            self.cleaned_df["customer_rating"].astype("Int64")
        )

        self.cleaned_df = self.cleaned_df.reset_index(drop=True)

    # --------------------------------------------------------------- outputs
    def get_cleaned_data(self):
        if self.cleaned_df is None:
            raise ValueError("Please clean the data first.")

        return self.cleaned_df

    def save_cleaned_data(self, output_path):
        """Write the cleaned data to a SEPARATE file (never the raw CSV)."""
        if self.cleaned_df is None:
            raise ValueError("Please clean the data first.")

        output_path = Path(output_path)
        if output_path.resolve() == Path(self.file_path).resolve():
            raise ValueError("Refusing to overwrite the raw input file.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.cleaned_df.to_csv(output_path, index=False)
        return output_path

    def show_report(self):
        if self.cleaned_df is None:
            raise ValueError("Please clean the data first.")

        print("\nMissing values before cleaning:")
        print(self.missing_report)

        print("\nDuplicate ticket IDs removed:")
        print(self.duplicate_count)

        print("\nInvalid values found:")
        print(self.invalid_report)

        print("\nRemaining missing values:")
        print(self.cleaned_df.isna().sum())
