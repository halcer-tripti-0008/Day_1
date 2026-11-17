import numpy as np
import pandas as pd


class TicketAnalytics:
    """Analytics on a cleaned ticket DataFrame.

    missing_report / invalid_report (optional) come from TicketDataProcessor so
    the report can say what was wrong BEFORE cleaning.
    """

    # A ticket still needs work in these statuses. "Unknown" or "Closed"
    # tickets are not counted as unresolved.
    UNRESOLVED_STATUSES = ("open", "pending", "in progress")

    def __init__(self, df, missing_report=None, invalid_report=None):
        self.df = df
        self.missing_report = missing_report
        self.invalid_report = invalid_report

    def _count_unresolved(self):
        status = (
            self.df["status"].astype(str).str.lower().str.replace("_", " ")
        )
        return int(status.isin(self.UNRESOLVED_STATUSES).sum())

    @staticmethod
    def _finite(series):
        """Numbers only: missing AND infinite values become NaN, so they are
        skipped by every average (an infinite value can never reach a mean)."""
        numbers = pd.to_numeric(series, errors="coerce").astype("float64")
        return numbers.where(np.isfinite(numbers))

    def _average_rating(self):
        mean = self._finite(self.df["customer_rating"]).mean()
        return None if pd.isna(mean) else round(float(mean), 2)

    def _average_resolution_by_category(self):
        times = self._finite(self.df["resolution_time_hours"])
        return times.groupby(self.df["category"]).mean().round(2).to_dict()

    def analyze(self):
        report = {
            "total_tickets": int(len(self.df)),

            "tickets_by_priority":
                {k: int(v) for k, v in self.df["priority"].value_counts().items()},

            "average_resolution_time_by_category":
                self._average_resolution_by_category(),

            "average_customer_rating": self._average_rating(),

            "unresolved_tickets": self._count_unresolved(),

            "missing_values":
                self.df.isna().sum().to_dict()
        }

        if self.missing_report is not None:
            report["missing_values_before_cleaning"] = self.missing_report

        if self.invalid_report is not None:
            report["invalid_records"] = self.invalid_report

        return report