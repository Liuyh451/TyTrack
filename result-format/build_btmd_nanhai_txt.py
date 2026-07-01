import csv
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np


NPY_PATH = Path("result_new_521.npy")
NPY_RECORDS_PATH = Path("forecast_records.txt")
NANHAI_RECORDS_PATH = Path("data/case/wpo/jangmi/forecast_records.txt")
OUTPUT_DIR = Path("btmd_nanhai_txt")
LEADS = [12, 24, 36, 48, 60, 72, 96, 120]
BTMD_MEMBER_INDEX = 8


def load_npy_report_times(path):
    text = path.read_text(encoding="utf-8")
    return re.findall(r"\[起报时间 \(BJ\)\]: ([^\n]+)", text)


def load_nanhai_records(path):
    records_by_report = OrderedDict()
    with path.open("r", encoding="utf-8", newline="") as file:
        for _ in range(5):
            next(file)
        reader = csv.DictReader(
            file,
            delimiter="\t",
            fieldnames=["report_time", "lead", "forecast_time", "lat", "lon", "pressure", "wind"],
        )
        for row in reader:
            report_time = row["report_time"]
            lead = int(row["lead"])
            records_by_report.setdefault(report_time, {})[lead] = {
                "pressure": row["pressure"],
                "wind": row["wind"],
            }
    return records_by_report


def output_name(report_time):
    stamp = report_time.replace("-", "").replace("T", "").replace(":", "")[:10]
    return f"2606_jangmi_btmd_nanhai_{stamp}.txt"


def main():
    result = np.load(NPY_PATH)
    npy_report_times = load_npy_report_times(NPY_RECORDS_PATH)
    nanhai_records = load_nanhai_records(NANHAI_RECORDS_PATH)
    selected_reports = list(nanhai_records.keys())[:4]

    OUTPUT_DIR.mkdir(exist_ok=True)

    for report_time in selected_reports:
        sample_index = npy_report_times.index(report_time)
        pressure_wind_by_lead = nanhai_records[report_time]
        lines = []

        for lead_index, lead in enumerate(LEADS):
            pressure_wind = pressure_wind_by_lead.get(lead)

            lat = result[sample_index, lead_index, BTMD_MEMBER_INDEX, 0]
            lon = result[sample_index, lead_index, BTMD_MEMBER_INDEX, 1]
            line = f"P+{lead}HR {lat:.1f}  {lon:.1f}"
            if pressure_wind is not None:
                line += f"  {pressure_wind['pressure']}  {pressure_wind['wind']}"
            lines.append(line)

        output_path = OUTPUT_DIR / output_name(report_time)
        output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        print(f"{output_path}: {len(lines)} lines")


if __name__ == "__main__":
    main()
