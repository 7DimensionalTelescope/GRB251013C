import pandas as pd
import yaml


# 1. Load the CSV data
df = pd.read_csv('/home/hongyp007/hongyp/GRB251013C/GRB_fitting/data/sdt_pivot.csv')

# 2. 지정된 경로의 YAML 파일 불러오기
yaml_path = '/home/hongyp007/hongyp/GRB251013C/GRB_fitting/data/7dt_filter_properties.yaml'
with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)

# 3. 'pivot' 섹션의 데이터를 매핑용 딕셔너리로 저장
pivot_map = config.get('pivot', {})

# 4. filter_name을 기준으로 wavelength 업데이트
# pivot_map에 해당 filter_name이 있는 경우만 값을 바꾸고, 없으면 기존 값을 유지합니다.
df['wavelength'] = df['filter_name'].map(pivot_map).fillna(df['wavelength'])

# 5. 결과 저장
df.to_csv('/home/hongyp007/hongyp/GRB251013C/GRB_fitting/data/sdt_pivot.csv', index=False)

print("Wavelengths updated successfully!")