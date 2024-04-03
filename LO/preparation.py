import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

def prepare_codes(df: pd.DataFrame):
    df.experience_level = pd.Categorical(df.experience_level)
    df = df.assign(employment_type=df.experience_level.cat.codes)

    df.employment_type = pd.Categorical(df.employment_type)
    df = df.assign(employment_type=df.employment_type.cat.codes)

    df.work_year = pd.Categorical(df.work_year)
    df = df.assign(work_year=df.work_year.cat.codes)

    df.salary_currency = pd.Categorical(df.salary_currency)
    df = df.assign(salary_currency=df.salary_currency.cat.codes)

    df.company_location = pd.Categorical(df.company_location)
    df = df.assign(company_location=df.company_location.cat.codes)

    df.company_size = pd.Categorical(df.company_size)
    df = df.assign(company_size=df.company_size.cat.codes)
    return df



if __name__ =='__main__':
    directory = Path('./data')
    input_path = directory / 'raw/ds_salaries.csv'
    output_path = directory / 'prepared/preprocessed_data.csv'

    df = pd.read_csv("data/raw/ds_salaries.csv")

    df = prepare_codes(df)

    output_path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(output_path)