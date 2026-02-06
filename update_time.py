import numpy as np
import os
from code.io import read_data
from code.const import TRIGGER_TIME,DATA_DIR
from datetime import datetime, timedelta
from astropy.time import Time

if __name__ == "__main__":
    filename = os.path.join(DATA_DIR, "raw_data.xlsx")
    df = read_data(filename)
    updated_time = []
    updated_t_diff = []
    updated_name = []
    updated_flux = []
    updated_error = []
    updated_limit = []
    
    for time, t_diff, name, val, error in zip(df["Starting Date"], df["T-T0"], df["Facility"], df["Mag"], df["Error"]):
        if isinstance(time, datetime):
            pass
        elif isinstance(time, float) and np.isfinite(time):
            time = datetime.fromisoformat(Time(time, format='jd', scale='utc').iso)
        elif isinstance(time, str):
            if "/" in time:
                time = datetime.strptime(time, '%Y/%m/%d %H:%M:%S')
            elif "T" in time:
                time = datetime.fromisoformat(time)
            else:
                time = datetime.fromisoformat(Time(float(time), format='jd', scale='utc').iso)
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

        if "\xa0" in name:
            name.replace("\xa0", "")
        else:
            name = name

        if isinstance(val, str) and ">" in val:
            val = float(val.replace(">", ""))
            updated_limit.append(True)
            error = 0
        else:
            val = float(val)
            updated_limit.append(False)
            error = float(error)
        
        updated_flux.append(val)
        updated_error.append(error)
        updated_name.append(name)
        updated_time.append(time)
        updated_t_diff.append((time - TRIGGER_TIME).total_seconds() / 3600.0)
        
    df["Starting Date"] = updated_time
    df["T-T0"] = updated_t_diff
    df["Facility"] = updated_name
    df["Mag"] = updated_flux
    df["Error"] = updated_error
    df["Limit"] = updated_limit
    df["T-T0"] = updated_t_diff
    df["T-T0"] = df["T-T0"].round(2)

    df.rename(columns={"T-T0": "Time"}, inplace=True)
    
    df.sort_values(by="Starting Date", inplace=True)
    
    df.to_excel(f"{DATA_DIR}/circular.xlsx", index=False)