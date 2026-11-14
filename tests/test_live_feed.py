"""Contract tests for the live MQTT boundary, independent of Caltrans access."""

import json
import gzip
import unittest
from datetime import UTC, datetime
from unittest.mock import patch

from app.live.consumer import validate_message
from app.live.feed import fresh_complete, parse_csv, parse_ftp_5min, validate_station_id
from app.live.publisher import fetch_ftp_5min


class LiveFeedTests(unittest.TestCase):
    SAMPLE = """station_id,timestamp,speed_mph,flow_veh_5min,occupancy_fraction,observed_percent
1001710,2026-11-05T12:00:00-08:00,55,120,0.08,100
1001710,2026-11-05T12:05:00-08:00,54,122,0.09,50
999,2026-11-05T12:00:00-08:00,50,90,0.1,100
"""

    def test_parse_and_filter_station_and_quality(self):
        rows = parse_csv(self.SAMPLE, "1001710")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].timestamp, "2026-11-05 12:00:00")
        fresh = fresh_complete(rows, now=datetime(2026, 11, 5, 20, 10, tzinfo=UTC))
        self.assertEqual([row.timestamp for row in fresh], ["2026-11-05 12:00:00"])
        self.assertEqual(fresh_complete(rows, now=datetime(2026, 11, 6, tzinfo=UTC)), [])

    def test_rejects_login_page_and_bad_values(self):
        with self.assertRaisesRegex(ValueError, "not station CSV"):
            parse_csv("<html>login</html>", "1001710")
        with self.assertRaisesRegex(ValueError, "invalid row"):
            parse_csv(self.SAMPLE.replace("0.08", "nan"), "1001710")
        with self.assertRaises(ValueError):
            validate_station_id("../1001710")

    def test_mqtt_contract(self):
        row = parse_csv(self.SAMPLE, "1001710")[0]
        self.assertEqual(validate_message(json.dumps(row.payload()).encode(), "1001710"), row)
        with self.assertRaisesRegex(ValueError, "wrong station"):
            validate_message(json.dumps(row.payload()).encode(), "999")
        corrupt = row.payload() | {"speed_mph": "fast"}
        with self.assertRaises(ValueError):
            validate_message(json.dumps(corrupt).encode(), "1001710")

    def test_documented_ftp_aggregate_columns(self):
        # VDS_ID,FLOW,OCCUPANCY,SPEED,VMT,VHT,Q,TRAVEL_TIME,DELAY,NUM_SAMPLES,PCT_OBSERVED
        file = gzip.compress(b"999,90,0.1,50,1,2,3,,0,4,100\n1108617,298,0.068,45.7,1,2,3,,0,4,100\n")
        rows = parse_ftp_5min(file, "5minagg_20261113061500.txt.gz", "1108617")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].timestamp, "2026-11-13 06:15:00")
        self.assertEqual((rows[0].flow_veh_5min, rows[0].occupancy_fraction, rows[0].speed_mph), (298, 0.068, 45.7))
        with self.assertRaisesRegex(ValueError, "missing"):
            parse_ftp_5min(file, "5minagg_20261113061500.txt.gz", "123")

    @patch("app.live.publisher.FTP_TLS")
    def test_ftp_does_not_redownload_same_timestamp(self, ftp_class):
        ftp = ftp_class.return_value
        filename = "5minagg_20261113061500.txt.gz"
        ftp.nlst.return_value = ["latest", filename]
        self.assertEqual(
            fetch_ftp_5min("pems.dot.ca.gov", "user", "password", 11, tls=True, after_filename=filename),
            (filename, None),
        )
        ftp.retrbinary.assert_not_called()


if __name__ == "__main__":
    unittest.main()
