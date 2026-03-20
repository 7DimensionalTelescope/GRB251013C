import pandas as pd

# 1. 필터별 파장 매핑 설정 (Angstrom)
wavelength_map = { 
    # 7DT Pivot Wavelengths & width
    'm400': [4013,250],
    'm425': [4255,250],
    'm450': [4508,250],
    'm475': [4753,250],
    'm500': [5003,250],
    'm525': [5248,250],
    'm550': [5501,250],
    'm575': [5749,250],
    'm600': [6001,250],
    'm625': [6248,250],
    'm650': [6501,250],
    'm675': [6745,250],
    'm700': [6999,250],
    'm725': [7246,250],
    'm750': [7489,250],
    'm775': [7752,250],
    'm800': [7992,250],
    'm825': [8240,250],
    'm850': [8483,250],
    'm875': [8729,250],
    
    # Clear band Pivot Wavelength & effective width (Sony ICS285AL - Calapai)
    'clear': [5622.12, 1876.90],
    # White band Pivot Wavelength & effective width (Pan-STARRS)
    'w': [6285.91, 2561.73],
    
    # SDSS pivot wavelengths & effective width (OAN-SPM/OPTICAM)
    # Sensor: Andor Zyla 4.2-Plus / https://www.southampton.ac.uk/opticam/project/instrument.page
    'u': [3565.44, 375.12],
    'g': [4806.36, 1071.48],
    "g'": [4806.36, 1071.48],
    'r': [6257.52, 1264.57],
    "r'": [6257.52, 1264.57],
    'i': [7580.91, 1288.67],
    'z': [8874.17, 1010.13],
    # Johnson-Cousins pivot wavelengths & effective width (OHP/Cam120)
    # Sensor: Andor Ikon L 936 / https://ohp.osupytheas.fr/le-telescope-de-120cm/
    'U': [3703.76, 457.14],
    'B': [4409.64, 949.22],
    'V': [5538.12, 1164.80],
    'R': [6533.95, 1676.68],
    'Rc': [6533.95, 1676.68],
    'I': [7989.40, 1337.00],
    'Ic': [7989.40, 1337.00],
    # 2MASS pivot wavelengths & effective width (2MASS)
    'J': [12350.00, 1520.26],
    'H': [16620.00, 2410.18],
    'K': [21590.00, 2506.19],
    'Ks': [21590.00, 2506.19],
    
    # GOTO pivot wavelengths & effective width
    'B_GOTO': [4622.88, 735.14],
    "G_GOTO": [5374.54, 857.73],
    'L_GOTO': [5523.78, 2162.29],
    'R_GOTO': [6426.84, 946.75],
    # Swift UVOT pivot wavelength & effective width (full transmission)
    'UVW1': [2628.38, 831.94],
    'UVW2': [2049.89, 346.13],
    'UVM2': [2248.59, 532.04],
    'u_swift': [3469.55, 658.46],
    'b_swift': [4350.30, 871.93],
    'v_swift': [5426.26, 659.82],
    'white': [3359.46, 3542.81],
    
    # SVOM VT central wavelengths & width
    'Blue': [5250, 2500], # 400 nm–650 nm
    'Red': [8250, 3000], # 650 nm–950 nm
}

# 2. CSV 파일 불러오기
df = pd.read_excel('/home/hongyp007/hongyp/GRB251013C/GRB_fitting/data/circular.xlsx').copy()

# Replace &apos; with ' in Filter column
df['Filter'] = df['Filter'].str.replace('&apos;', "'", regex=False)

# Create separate mapping dictionaries for wavelength and width
wave_dict = {k: v[0] for k, v in wavelength_map.items()}
width_dict = {k: v[1] for k, v in wavelength_map.items()}

# Map the values to the new columns
df['wavelength'] = df['Filter'].map(wave_dict)
df['filter_width'] = df['Filter'].map(width_dict)

# 4. 결과 확인 및 저장
print(df.head())
df.to_csv('/home/hongyp007/hongyp/GRB251013C/GRB_fitting/data/circular_wavelength.csv', index=False)