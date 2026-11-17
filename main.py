import sys
from pathlib import Path

from ticket_processor import TicketDataProcessor, TicketDataError
from ticket_analytics import TicketAnalytics

BASE_DIR = Path(__file__).resolve().parent  # works from any working directory


def main():
    processor = TicketDataProcessor(BASE_DIR / "tickets.csv")

    try:
        processor.load_data()
        print("CSV loaded successfully!")
        processor.validate_columns()
        print("Column validation successful!")
        cleaned_data = processor.clean_data()
        print("Data cleaning completed successfully!")
    except (TicketDataError, FileNotFoundError) as error:
        print(f"\nERROR: {error}")
        return 1

    print("\nCleaned Dataset:")
    print(cleaned_data)

    processor.show_report()

    output_file = processor.save_cleaned_data(BASE_DIR / "output" / "cleaned_tickets.csv")
    print(f"\nCleaned dataset saved successfully to {output_file}")

    analytics = TicketAnalytics(
        cleaned_data,
        missing_report=processor.missing_report,
        invalid_report=processor.invalid_report,
    )
    report = analytics.analyze()

    print("\nTicket Analytics Report:")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())