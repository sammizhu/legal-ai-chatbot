import pandas as pd
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


INPUT_DTA_PATH = "V3DistrictJudgesBIO.dta"
OUTPUT_JSON_PATH = "judge_profiles.json"

df = pd.read_stata(INPUT_DTA_PATH)

def stata_date_to_str(val):
    if pd.isnull(val):
        return "Unknown"
    return (datetime(1960, 1, 1) + timedelta(days=int(val))).strftime("%Y-%m-%d")

date_fields = ["start_date", "senior_date", "end_date"]
for field in date_fields:
    df[field] = df[field].apply(stata_date_to_str)

header_mapping = {
    "songername": "Name",
    "district": "Judicial district",
    "Courthouse": "Courthouse",
    "start_date": "Began service",
    "senior_date": "Assume senior status",
    "end_date": "Ended service",
    "x_dem": "Democratic party affiliation",
    "x_repub": "Republican party affiliation",
    "x_insba": "Member of the state bar association",
    "x_elev": "Elevated from a lower court position",
    "x_unity": "Political unity measure or bipartisan indicator",
    "x_aba": "Rated by the American Bar Association",
    "x_crossa": "Cross-appointed by presidents of different parties",
    "x_pfed": "Previous experience in the federal government",
    "x_pothfe": "Other federal executive experience",
    "x_plprof": "Law professor prior to judgeship",
    "x_pscab": "Served in a state cabinet",
    "x_pcab": "Served in a president's cabinet",
    "x_pusa": "Served as U.S. Attorney",
    "x_pssen": "Served as a state senator",
    "x_paag": "Served as assistant attorney general",
    "x_psp": "Served in state police or public safety",
    "x_pslc": "Served in state legislature (lower chamber)",
    "x_pssc": "Served in state legislature (senate chamber)",
    "x_pshou": "Served in U.S. House of Representatives",
    "x_psg": "Served in state government general positions",
    "x_psgo": "Served as a state governor",
    "x_psen": "Served in the U.S. Senate",
    "x_psat": "Served as state attorney general",
    "x_ppriv": "Worked in private legal practice",
    "x_pmayor": "Served as a city mayor",
    "x_plocct": "Held a local city/county role",
    "x_phous": "Served in U.S. House",
    "x_pgov": "Served as a state or federal governor",
    "x_pda": "Served as a district attorney",
    "x_pcc": "Served on a city council",
    "x_pccoun": "Served on a county council",
    "x_pausa": "Served as assistant U.S. attorney",
    "x_pasat": "Served as assistant state attorney general",
    "x_pag": "Served as attorney general",
    "x_pada": "Assistant district attorney experience",
    "x_pgovt": "General government service",
    "x_llmsjd": "Earned a JD or LLM degree",
    "x_prot": "Protestant religious affiliation",
    "x_evang": "Evangelical Christian affiliation",
    "x_mline": "Mainline Protestant affiliation",
    "x_norel": "No religious affiliation",
    "x_cath": "Catholic religious affiliation",
    "x_jew": "Jewish religious affiliation",
    "x_black": "Identifies as Black",
    "x_nw": "Identifies as non-white",
    "x_fem": "Identifies as female",
    "x_jdpub": "Earned JD from a public university",
    "x_bapub": "Earned BA from a public university",
    "x_b10s": "Born in 1910s",
    "x_b20s": "Born in 1920s",
    "x_b30s": "Born in 1930s",
    "x_b40s": "Born in 1940s",
    "x_b50s": "Born in 1950s",
    "x_pbank": "Worked in banking or finance sector",
    "x_pmag": "Served as a magistrate judge",
    "x_ageon40s": "Active on bench in their 40s",
    "x_ageon50s": "Active on bench in their 50s",
    "x_ageon60s": "Active on bench in their 60s",
    "x_ageles40": "Became judge before age 40",
    "x_agemor70": "Remained judge past age 70",
    "x_pago": "Served as Inspector General or equivalent",
    "retirementfromactiveservice": "Retired from",
    "start_year": "Year started judicial service",
    "senior_year": "Year entered senior status",
    "end_year": "Year ended judicial service"
}
df = df.rename(columns=header_mapping)

def normalize(val):
    if pd.isnull(val):
        return "Unknown"
    elif isinstance(val, float) and val in [0.0, 1.0]:
        return "Yes" if val == 1.0 else "No"
    return val

df = df.applymap(normalize)

## ANALYSIS
df.to_json(OUTPUT_JSON_PATH, orient="records", indent=2)
print(f"Saved cleaned JSON to {OUTPUT_JSON_PATH}")

with open(OUTPUT_JSON_PATH, 'r') as f:
    data = json.load(f)

# Extract the "Began service" years
years = []
for record in data:
    began_service = record.get("Began service", "Unknown")
    if began_service != "Unknown":
        try:
            year = int(began_service.split("-")[0])  # Extract the year
            years.append(year)
        except ValueError:
            pass  # Skip invalid dates

# Create the histogram
plt.hist(years, bins=range(min(years), max(years) + 1), edgecolor='black')
plt.title("Histogram of Began Service Years")
plt.xlabel("Year")
plt.ylabel("Frequency")

# Save the histogram as an image
plt.savefig("began_service_years_histogram.png")
print("Histogram saved as 'began_service_years_histogram.png'")
plt.show()