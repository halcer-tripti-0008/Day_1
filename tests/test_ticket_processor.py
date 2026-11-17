import hashlib
from pathlib import Path

import pandas as pd
import pytest

from ticket_analytics import TicketAnalytics
from ticket_processor import (
    EmptyDatasetError,
    InvalidCSVError,
    MissingColumnsError,
    TicketDataProcessor,
)

HEADER = "ticket_id,category,priority,description,resolution_time_hours,customer_rating,status\n"
SAMPLE_CSV = Path(__file__).resolve().parent.parent / "tickets.csv"


def write_csv(tmp_path, body, header=HEADER):
    path = tmp_path / "tickets_test.csv"
    path.write_text(header + body)
    return path


def run(path, **kwargs):
    processor = TicketDataProcessor(path, **kwargs)
    processor.load_data()
    processor.validate_columns()
    processor.clean_data()
    return processor


# 1. valid input ------------------------------------------------------------
def test_valid_input_is_cleaned_and_typed(tmp_path):
    path = write_csv(
        tmp_path,
        "1,Billing,High,Charged twice,4.5,4,Resolved\n"
        "2,Account,Low,Change email,2,5,Open\n",
    )
    processor = run(path)
    df = processor.get_cleaned_data()
    assert len(df) == 2
    assert str(df["resolution_time_hours"].dtype) == "float64"
    assert str(df["customer_rating"].dtype) == "Int64"
    assert processor.duplicate_count == 0
    assert processor.invalid_report["invalid_resolution_time_hours"] == 0


# 2. invalid CSV --------------------------------------------------------------
def test_malformed_csv_raises_invalid_csv_error(tmp_path):
    path = write_csv(tmp_path, "1,Billing,High,desc,4.5,4,Resolved,EXTRA,FIELDS\n")
    with pytest.raises(InvalidCSVError):
        TicketDataProcessor(path).load_data()


def test_ragged_row_raises_invalid_csv_error(tmp_path):
    path = write_csv(
        tmp_path,
        "1,Billing,High,desc,4.5,4,Resolved\n2,Account,Low,desc,1,5,Open,EXTRA\n",
    )
    with pytest.raises(InvalidCSVError):
        TicketDataProcessor(path).load_data()


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        TicketDataProcessor(tmp_path / "nope.csv").load_data()


def test_binary_file_raises_invalid_csv_error(tmp_path):
    path = tmp_path / "binary.csv"
    path.write_bytes(b"\xff\xfe\x00\x81\x82\x83")
    with pytest.raises(InvalidCSVError):
        TicketDataProcessor(path).load_data()


# 3. empty dataset ------------------------------------------------------------
def test_zero_byte_file_raises_empty_dataset_error(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(EmptyDatasetError):
        TicketDataProcessor(path).load_data()


def test_header_only_file_raises_empty_dataset_error(tmp_path):
    with pytest.raises(EmptyDatasetError):
        TicketDataProcessor(write_csv(tmp_path, "")).load_data()


# 4. missing required columns ------------------------------------------------------
def test_missing_columns_are_reported_clearly(tmp_path):
    path = tmp_path / "cols.csv"
    path.write_text("ticket_id,category,priority\n1,Billing,High\n")
    processor = TicketDataProcessor(path)
    processor.load_data()
    with pytest.raises(MissingColumnsError) as error:
        processor.validate_columns()
    assert error.value.missing_columns == [
        "description", "resolution_time_hours", "customer_rating", "status",
    ]
    assert "description" in str(error.value)


def test_validate_before_load_gives_clear_message():
    with pytest.raises(ValueError, match="load the CSV first"):
        TicketDataProcessor("whatever.csv").validate_columns()


# 5. duplicate ticket IDs ---------------------------------------------------------
def test_duplicate_ids_keep_first_and_are_counted(tmp_path):
    path = write_csv(
        tmp_path,
        "1,Billing,High,first copy,4.0,4,Resolved\n"
        "1,Billing,High,second copy,9.0,2,Closed\n"
        "2,Account,Low,other,1.0,5,Open\n",
    )
    processor = run(path)
    df = processor.get_cleaned_data()
    assert processor.duplicate_count == 1
    assert list(df["description"]) == ["first copy", "other"]


# 6. missing / invalid values ----------------------------------------------------------
def test_invalid_and_missing_values_follow_the_rules(tmp_path):
    path = write_csv(
        tmp_path,
        "1,Billing,High,ok,abc,3,Resolved\n"          # non-numeric time
        "2,Billing,High,ok,-5,9,Resolved\n"           # negative time, rating > 5
        "3,,low,,2.5,3.5,  open  \n"                  # blanks, lower case, float rating
        "4,Account,Low,ok,1,,Open\n"                  # blank rating (just missing)
        ",Account,Low,ok,1,1,Open\n",                 # no ticket_id -> removed
    )
    processor = run(path)
    df = processor.get_cleaned_data().set_index("ticket_id")

    assert len(df) == 4
    assert pd.isna(df.loc[1, "resolution_time_hours"])
    assert pd.isna(df.loc[2, "resolution_time_hours"]) and pd.isna(df.loc[2, "customer_rating"])
    assert df.loc[3, "category"] == "Unknown" and df.loc[3, "description"] == "Unknown"
    assert df.loc[3, "priority"] == "Low" and df.loc[3, "status"] == "Open"  # normalised
    assert pd.isna(df.loc[3, "customer_rating"])                            # 3.5 not whole
    assert df.loc[3, "resolution_time_hours"] == 2.5                         # valid untouched

    report = processor.invalid_report
    assert report["missing_ticket_ids_removed"] == 1
    assert report["invalid_resolution_time_hours"] == 1
    assert report["negative_resolution_time"] == 1
    assert report["invalid_customer_rating"] == 2  # the 9 and the 3.5, NOT the blank


def test_missing_numbers_are_not_imputed_by_default(tmp_path):
    path = write_csv(tmp_path, "1,Billing,High,ok,,,Open\n2,Billing,Low,ok,4,4,Open\n")
    df = run(path).get_cleaned_data()
    assert df["resolution_time_hours"].isna().sum() == 1
    assert df["customer_rating"].isna().sum() == 1


def test_median_imputation_when_enabled(tmp_path):
    path = write_csv(
        tmp_path,
        "1,Billing,High,ok,,,Open\n2,Billing,Low,ok,4,4,Open\n3,Billing,Low,ok,6,2,Open\n",
    )
    df = run(path, impute_missing=True).get_cleaned_data()
    assert df.loc[0, "resolution_time_hours"] == 5.0   # median of 4 and 6
    assert df.loc[0, "customer_rating"] == 3           # median of 4 and 2
    assert df.isna().sum().sum() == 0


# 7. raw data integrity -----------------------------------------------------------------
def test_raw_file_is_not_modified_and_cannot_be_overwritten(tmp_path):
    path = write_csv(tmp_path, "1,Billing,High,a,1,1,Open\n1,Billing,High,b,2,2,Open\n")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    processor = run(path)
    saved = processor.save_cleaned_data(tmp_path / "out" / "cleaned.csv")
    assert saved.exists()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert len(processor.df) == 2  # the loaded raw frame still has both rows
    with pytest.raises(ValueError, match="overwrite"):
        processor.save_cleaned_data(path)


# 8. analytics ----------------------------------------------------------------------------
def test_analytics_match_hand_calculation(tmp_path):
    # priorities: High x2, Low x1 | Billing times 4 and 8 -> 6.0, Account 2 -> 2.0
    # ratings 5, 3, missing -> 4.0 | statuses Resolved, Open, In Progress -> 2 unresolved
    path = write_csv(
        tmp_path,
        "1,Billing,High,a,4,5,Resolved\n"
        "2,Billing,High,b,8,3,Open\n"
        "3,Account,Low,c,2,,In Progress\n",
    )
    processor = run(path)
    report = TicketAnalytics(
        processor.get_cleaned_data(), processor.missing_report, processor.invalid_report
    ).analyze()
    assert report["tickets_by_priority"] == {"High": 2, "Low": 1}
    assert report["average_resolution_time_by_category"] == {"Account": 2.0, "Billing": 6.0}
    assert report["average_customer_rating"] == 4.0
    assert report["unresolved_tickets"] == 2
    assert report["missing_values"]["customer_rating"] == 1
    assert report["missing_values_before_cleaning"]["customer_rating"] == 1


def test_analytics_on_provided_tickets_csv_manually_verified():
    processor = run(SAMPLE_CSV)
    report = TicketAnalytics(
        processor.get_cleaned_data(), processor.missing_report, processor.invalid_report
    ).analyze()
    assert report["total_tickets"] == 19                    # 20 rows - duplicate ID 119
    assert processor.duplicate_count == 1
    assert report["tickets_by_priority"] == {"High": 7, "Medium": 5, "Low": 5, "Critical": 2}
    assert report["unresolved_tickets"] == 5                # Open 107,112,117 + Pending 109,114
    assert report["average_customer_rating"] == 3.67        # 66 / 18 by hand
    assert report["average_resolution_time_by_category"]["Billing"] == 3.33  # 20.0 / 6 by hand
    assert report["invalid_records"]["invalid_resolution_time_hours"] == 1   # 'invalid' in 110


def test_analytics_with_all_values_missing(tmp_path):
    df = run(write_csv(tmp_path, "1,Billing,Low,x,,,Open\n")).get_cleaned_data()
    report = TicketAnalytics(df).analyze()
    assert report["average_customer_rating"] is None
