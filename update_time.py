import numpy as np
import os
from code.io import read_circular_data
from code.const import TRIGGER_TIME,DATA_DIR
from datetime import datetime, timedelta
from astropy.time import Time

if __name__ == "__main__":
    filename = os.path.join(DATA_DIR, "raw_data.xlsx")
    df = read_circular_data(filename)
    updated_time = []
    updated_t_diff = []
    for time, t_diff in zip(df["Starting Date"], df["T-T0"]):
        if isinstance(time, datetime):
            pass
        elif isinstance(time, float) and np.isfinite(time):
            time = datetime.fromisoformat(Time(time, format='jd', scale='utc').iso)

        else:
            if "days" in t_diff:
                time = TRIGGER_TIME + timedelta(days=float(t_diff.replace("days", "")))
            elif "hr" in t_diff:
                time = TRIGGER_TIME + timedelta(hours=float(t_diff.replace("hr", "")))
            elif "min" in t_diff:
                time = TRIGGER_TIME + timedelta(minutes=float(t_diff.replace("min", "")))
            elif "sec" in t_diff:
                time = TRIGGER_TIME + timedelta(seconds=float(t_diff.replace("sec", "")))
            else:
                raise ValueError(f"Unknown time format: {t_diff}")
        updated_time.append(time)
        updated_t_diff.append((time - TRIGGER_TIME).total_seconds() / 3600.0)
        
    df["Starting Date"] = updated_time
    df["T-T0"] = updated_t_diff
    
    df["T-T0"] = updated_t_diff
    df["T-T0"] = df["T-T0"].round(2)

    df.rename(columns={"T-T0": "Time"}, inplace=True)

    df.sort_values(by="Starting Date", inplace=True)
    
    df.to_excel(f"{DATA_DIR}/circular.xlsx", index=False)