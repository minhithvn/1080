# ============================================
# PHẦN 1: IMPORTS VÀ SETUP CƠ BẢN
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import time
import hashlib
import json
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')

# Cấu hình trang
st.set_page_config(
    page_title="Phân Tích Cổ Phiếu VN",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS cho giao diện đẹp hơn
st.markdown("""
<style>
.big-font {font-size:20px !important; font-weight:bold;}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
    padding: 1.5rem; 
    border-radius: 10px; 
    color: white;
}
.buy-signal {
    background: linear-gradient(135deg, #00c853 0%, #00e676 100%); 
    color: white; 
    padding: 15px; 
    border-radius: 10px; 
    font-weight: bold; 
    text-align: center; 
    box-shadow: 0 4px 15px rgba(0,200,83,0.4);
}
.sell-signal {
    background: linear-gradient(135deg, #d32f2f 0%, #f44336 100%); 
    color: white; 
    padding: 15px; 
    border-radius: 10px; 
    font-weight: bold; 
    text-align: center; 
    box-shadow: 0 4px 15px rgba(211,47,47,0.4);
}
.hold-signal {
    background: linear-gradient(135deg, #ff9800 0%, #ffc107 100%); 
    color: white; 
    padding: 15px; 
    border-radius: 10px; 
    font-weight: bold; 
    text-align: center; 
    box-shadow: 0 4px 15px rgba(255,152,0,0.4);
}
.prediction-box {
    background: linear-gradient(135deg, #1e88e5 0%, #42a5f5 100%); 
    padding: 20px; 
    border-radius: 10px; 
    color: white; 
    margin: 10px 0;
}
.money-flow-box {
    background: linear-gradient(135deg, #9c27b0 0%, #ba68c8 100%); 
    padding: 20px; 
    border-radius: 10px; 
    color: white; 
    margin: 10px 0;
}
.sector-card {
    background: linear-gradient(135deg, #3f51b5 0%, #5c6bc0 100%);
    padding: 15px;
    border-radius: 8px;
    color: white;
    margin: 5px 0;
}
.footer-stats {
    position: fixed;
    bottom: 10px;
    right: 10px;
    background: rgba(0,0,0,0.95);
    color: white;
    padding: 15px 20px;
    border-radius: 12px;
    font-size: 12px;
    z-index: 999;
    box-shadow: 0 8px 32px rgba(0,0,0,0.6);
    border: 1px solid rgba(255,255,255,0.1);
}
</style>
""", unsafe_allow_html=True)
# ============================================
# PHẦN 2A: DATABASE MÃ CỔ PHIẾU A-D (Nhóm 1/5)
# ============================================

# Danh sách mã A-D
STOCKS_A_TO_D = [
    'AAA', 'AAM', 'AAS', 'AAT', 'AAV', 'ABB', 'ABC', 'ABR', 'ABS', 'ABT', 'ABW', 'ACC', 'ACB',
    'ACE', 'ACG', 'ACL', 'ACM', 'ACS', 'ACV', 'ADC', 'ADP', 'ADS', 'AEC', 'AFX', 'AG1', 'AGE',
    'AGF', 'AGG', 'AGM', 'AGP', 'AGR', 'AIC', 'ALT', 'ALV', 'AMC', 'AME', 'AMP', 'AMS', 'AMV',
    'ANT', 'ANV', 'APC', 'APF', 'APG', 'APH', 'API', 'APL', 'APP', 'APS', 'APT', 'AQN', 'ARM',
    'ART', 'ASA', 'ASG', 'ASM', 'ASP', 'AST', 'ATA', 'ATB', 'ATG', 'ATS', 'AUM',
    'B82', 'BAB', 'BAF', 'BAL', 'BAX', 'BBC', 'BCE', 'BCF', 'BCG', 'BCI', 'BCM', 'BCP', 'BCV',
    'BDB', 'BDC', 'BDF', 'BDG', 'BDT', 'BDW', 'BED', 'BEL', 'BFC', 'BFT', 'BGW', 'BHA', 'BHC',
    'BHG', 'BHI', 'BHK', 'BHN', 'BHP', 'BHT', 'BHV', 'BIC', 'BID', 'BIG', 'BII', 'BIO', 'BKC',
    'BKG', 'BLF', 'BLI', 'BLN', 'BLT', 'BLW', 'BMC', 'BMD', 'BMF', 'BMG', 'BMI', 'BMJ', 'BMN',
    'BMP', 'BMS', 'BMV', 'BMW', 'BMX', 'BNA', 'BNW', 'BOT', 'BPC', 'BPH', 'BPP', 'BPW', 'BRC',
    'BRR', 'BRS', 'BRT', 'BSA', 'BSC', 'BSD', 'BSG', 'BSH', 'BSI', 'BSL', 'BSP', 'BSQ', 'BSR',
    'BST', 'BT1', 'BT6', 'BTA', 'BTC', 'BTD', 'BTG', 'BTH', 'BTN', 'BTP', 'BTR', 'BTS', 'BTT',
    'BTV', 'BTW', 'BVB', 'BVG', 'BVL', 'BVN', 'BVS', 'BWA', 'BWE', 'BWS', 'BXH',
    'C12', 'C21', 'C22', 'C32', 'C36', 'C4G', 'C47', 'C69', 'C71', 'C92', 'CAD', 'CAG', 'CAN',
    'CAP', 'CAT', 'CAV', 'CBC', 'CBB', 'CC1', 'CC4', 'CCA', 'CCI', 'CCL', 'CCM', 'CCP', 'CCR',
    'CCV', 'CDC', 'CDH', 'CDN', 'CDO', 'CDP', 'CDR', 'CE1', 'CEN', 'CEO', 'CER', 'CET', 'CFC',
    'CFM', 'CFV', 'CGV', 'CH5', 'CHP', 'CHR', 'CHS', 'CIA', 'CIC', 'CID', 'CIG', 'CII', 'CIP',
    'CJC', 'CKA', 'CKD', 'CKG', 'CKV', 'CLC', 'CLE', 'CLG', 'CLL', 'CLM', 'CLP', 'CLW', 'CLX',
    'CM8', 'CMA', 'CMC', 'CMD', 'CMF', 'CMG', 'CMI', 'CMK', 'CMM', 'CMN', 'CMP', 'CMS', 'CMT',
    'CMV', 'CMW', 'CMX', 'CNC', 'CNG', 'CNN', 'CNT', 'COM', 'CPA', 'CPC', 'CPH', 'CPI', 'CRC',
    'CRE', 'CSC', 'CSG', 'CSI', 'CSM', 'CSV', 'CT3', 'CT6', 'CTA', 'CTB', 'CTC', 'CTD', 'CTF',
    'CTG', 'CTI', 'CTM', 'CTN', 'CTP', 'CTR', 'CTS', 'CTT', 'CTV', 'CTW', 'CTX', 'CVN', 'CVP',
    'CVT', 'CX8', 'CYC',
    'D11', 'D2D', 'DAC', 'DAD', 'DAE', 'DAG', 'DAH', 'DAN', 'DAP', 'DAS', 'DAT', 'DBC', 'DBD',
    'DBM', 'DBT', 'DC1', 'DC2', 'DC4', 'DCC', 'DCF', 'DCG', 'DCH', 'DCI', 'DCL', 'DCM', 'DCR',
    'DCS', 'DCT', 'DDG', 'DDM', 'DDN', 'DDV', 'DFC', 'DGC', 'DGT', 'DGW', 'DHA', 'DHB', 'DHC',
    'DHG', 'DHI', 'DHM', 'DHN', 'DHP', 'DHS', 'DHT', 'DIC', 'DID', 'DIG', 'DIH', 'DKC', 'DKH',
    'DL1', 'DLD', 'DLG', 'DLM', 'DLR', 'DLT', 'DLV', 'DMC', 'DNA', 'DNC', 'DND', 'DNE', 'DNH',
    'DNL', 'DNM', 'DNN', 'DNP', 'DNR', 'DNS', 'DNT', 'DNW', 'DOC', 'DOP', 'DPC', 'DPG', 'DPH',
    'DPM', 'DPP', 'DPR', 'DPS', 'DQC', 'DRC', 'DRG', 'DRH', 'DRI', 'DRL', 'DS3', 'DSC', 'DSD',
    'DSG', 'DSN', 'DSP', 'DST', 'DSV', 'DT4', 'DTA', 'DTB', 'DTC', 'DTD', 'DTE', 'DTG', 'DTH',
    'DTI', 'DTK', 'DTL', 'DTN', 'DTP', 'DTT', 'DTV', 'DVG', 'DVM', 'DVN', 'DVP', 'DVW', 'DWC',
    'DWS', 'DXG', 'DXL', 'DXP', 'DXS', 'DXV', 'DZM'
]
# ============================================
# PHẦN 2B: DATABASE MÃ CỔ PHIẾU E-L (Nhóm 2/5)
# ============================================

STOCKS_E_TO_L = [
    'E12', 'E29', 'EBS', 'ECI', 'EFI', 'EIB', 'EIC', 'EID', 'EIN', 'ELC', 'EMC', 'EME', 'EVE',
    'EVF', 'EVG', 'EVS',
    'FCM', 'FCN', 'FCS', 'FDC', 'FGL', 'FHS', 'FIC', 'FID', 'FIR', 'FIT', 'FLC', 'FMC', 'FOC',
    'FOX', 'FPT', 'FRC', 'FRM', 'FRT', 'FSO', 'FTI', 'FTM', 'FTS',
    'G20', 'G36', 'GAB', 'GAS', 'GBS', 'GDT', 'GDW', 'GEE', 'GEG', 'GER', 'GEX', 'GGG', 'GHC',
    'GIC', 'GIL', 'GKM', 'GLT', 'GLW', 'GMC', 'GMD', 'GMH', 'GMP', 'GMS', 'GMX', 'GSM', 'GSP',
    'GTA', 'GTC', 'GTD', 'GTH', 'GTN', 'GTS', 'GTT', 'GVR', 'GVT',
    'HAD', 'HAG', 'HAH', 'HAI', 'HAM', 'HAP', 'HAR', 'HAS', 'HAT', 'HAV', 'HAX', 'HBB', 'HBC',
    'HBD', 'HBH', 'HBS', 'HCD', 'HCM', 'HCT', 'HDB', 'HDC', 'HDG', 'HDM', 'HDO', 'HDP', 'HEJ',
    'HEM', 'HEP', 'HES', 'HFC', 'HFX', 'HGM', 'HGT', 'HGW', 'HHA', 'HHC', 'HHG', 'HHL', 'HHN',
    'HHP', 'HHR', 'HHS', 'HHV', 'HID', 'HIG', 'HII', 'HJC', 'HJS', 'HKB', 'HKP', 'HKT', 'HLA',
    'HLB', 'HLC', 'HLD', 'HLG', 'HLR', 'HLS', 'HLT', 'HLY', 'HMC', 'HMG', 'HMH', 'HMR', 'HMS',
    'HNA', 'HNB', 'HND', 'HNE', 'HNF', 'HNG', 'HNI', 'HNM', 'HNN', 'HNP', 'HNR', 'HNX', 'HOA',
    'HOT', 'HPC', 'HPD', 'HPG', 'HPH', 'HPI', 'HPM', 'HPP', 'HPR', 'HPS', 'HPT', 'HPW', 'HPX',
    'HQC', 'HRC', 'HRG', 'HRT', 'HSA', 'HSC', 'HSG', 'HSI', 'HSL', 'HSM', 'HSP', 'HST', 'HSV',
    'HT1', 'HT3', 'HTA', 'HTB', 'HTC', 'HTG', 'HTI', 'HTL', 'HTM', 'HTN', 'HTP', 'HTR', 'HTS',
    'HTT', 'HTV', 'HTW', 'HU1', 'HU3', 'HU4', 'HU6', 'HUB', 'HUG', 'HUT', 'HVA', 'HVG', 'HVH',
    'HVN', 'HVT', 'HVX',
    'ICF', 'ICI', 'ICN', 'ICT', 'IDC', 'IDI', 'IDJ', 'IDV', 'IFC', 'IFS', 'IHK', 'IJC', 'ILA',
    'ILB', 'ILS', 'IME', 'IMP', 'IMS', 'INC', 'INN', 'IRC', 'ISG', 'ISH', 'IST', 'ITA', 'ITC',
    'ITD', 'ITQ', 'ITS', 'IVS',
    'JOS', 'JVC',
    'KAC', 'KBC', 'KCB', 'KCE', 'KCP', 'KDC', 'KDH', 'KDM', 'KGM', 'KGU', 'KHA', 'KHB', 'KHG',
    'KHL', 'KHP', 'KHS', 'KIP', 'KKC', 'KLB', 'KLF', 'KLS', 'KMR', 'KMT', 'KOS', 'KPF', 'KSA',
    'KSB', 'KSC', 'KSE', 'KSF', 'KSH', 'KSK', 'KSQ', 'KSS', 'KST', 'KSV', 'KTC', 'KTL', 'KTS',
    'KTT', 'KVC',
    'L10', 'L14', 'L18', 'L40', 'L43', 'L44', 'L45', 'L61', 'L62', 'L63', 'LAF', 'LAS', 'LBE',
    'LBM', 'LCC', 'LCG', 'LCM', 'LCS', 'LCW', 'LDP', 'LDG', 'LEC', 'LGC', 'LGL', 'LHC', 'LHG',
    'LIG', 'LIX', 'LM3', 'LM7', 'LM8', 'LMH', 'LO5', 'LPB', 'LPT', 'LSS', 'LTC', 'LTG', 'LUT'
]
# ============================================
# PHẦN 2C: DATABASE MÃ CỔ PHIẾU M-P (Nhóm 3/5)
# ============================================

STOCKS_M_TO_P = [
    'MAC', 'MAS', 'MBB', 'MBC', 'MBG', 'MBN', 'MBS', 'MCC', 'MCF', 'MCG', 'MCH', 'MCI', 'MCM',
    'MCO', 'MCP', 'MCS', 'MCV', 'MDC', 'MDF', 'MDG', 'MED', 'MEL', 'MFS', 'MGC', 'MGG', 'MHC',
    'MHL', 'MHP', 'MIC', 'MIE', 'MIG', 'MIL', 'MIM', 'MIT', 'MJC', 'MKP', 'MKT', 'MKV', 'MLC',
    'MLS', 'MLT', 'MMC', 'MML', 'MNC', 'MND', 'MSB', 'MSC', 'MSH', 'MSN', 'MSR', 'MST', 'MTA',
    'MTB', 'MTC', 'MTG', 'MTH', 'MTL', 'MTP', 'MTS', 'MTV', 'MVB', 'MVC', 'MVN', 'MWG',
    'NAB', 'NAF', 'NAG', 'NAP', 'NAS', 'NAV', 'NBB', 'NBC', 'NBE', 'NBP', 'NBT', 'NBW', 'NCB',
    'NCP', 'NCS', 'NCT', 'NDC', 'NDF', 'NDN', 'NDP', 'NDT', 'NDW', 'NDX', 'NED', 'NET', 'NFC',
    'NGC', 'NHA', 'NHC', 'NHH', 'NHN', 'NHP', 'NHS', 'NHT', 'NHV', 'NHW', 'NIS', 'NJC', 'NKD',
    'NKG', 'NLC', 'NLG', 'NLS', 'NMC', 'NNC', 'NNG', 'NNT', 'NPS', 'NQB', 'NQN', 'NQT', 'NRC',
    'NSC', 'NSG', 'NSH', 'NSL', 'NSN', 'NSS', 'NST', 'NT2', 'NTB', 'NTC', 'NTF', 'NTH', 'NTL',
    'NTP', 'NTT', 'NTW', 'NUE', 'NVB', 'NVL', 'NVN', 'NVP', 'NVT', 'NWT',
    'OCB', 'OCH', 'ODE', 'OGC', 'OIL', 'ONW', 'OPC', 'OPS', 'ORS',
    'PAC', 'PAI', 'PAN', 'PAP', 'PAT', 'PBC', 'PBP', 'PBS', 'PBT', 'PC1', 'PCC', 'PCE', 'PCF',
    'PCG', 'PCH', 'PCM', 'PCN', 'PCS', 'PCT', 'PDC', 'PDN', 'PDR', 'PDT', 'PEC', 'PEG', 'PEN',
    'PET', 'PFL', 'PGB', 'PGC', 'PGD', 'PGI', 'PGN', 'PGS', 'PGT', 'PGV', 'PHC', 'PHH', 'PHN',
    'PHP', 'PHR', 'PHS', 'PHT', 'PIA', 'PIC', 'PID', 'PIS', 'PIT', 'PIV', 'PJC', 'PJS', 'PJT',
    'PKR', 'PLA', 'PLC', 'PLE', 'PLO', 'PLP', 'PLX', 'PMB', 'PMC', 'PME', 'PMG', 'PMJ', 'PMP',
    'PMS', 'PMT', 'PMW', 'PNC', 'PNG', 'PNJ', 'PNP', 'PNT', 'POM', 'POS', 'POT', 'POW', 'PPC',
    'PPE', 'PPG', 'PPH', 'PPI', 'PPP', 'PPS', 'PPT', 'PPY', 'PRC', 'PRE', 'PRT', 'PRW', 'PSB',
    'PSC', 'PSD', 'PSE', 'PSG', 'PSH', 'PSI', 'PSL', 'PSN', 'PSP', 'PSW', 'PTC', 'PTD', 'PTE',
    'PTG', 'PTH', 'PTI', 'PTL', 'PTO', 'PTP', 'PTQ', 'PTS', 'PTT', 'PTV', 'PV2', 'PVA', 'PVB',
    'PVC', 'PVD', 'PVE', 'PVG', 'PVH', 'PVI', 'PVL', 'PVM', 'PVP', 'PVR', 'PVS', 'PVT', 'PVV',
    'PVX', 'PVY', 'PWA', 'PWS', 'PX1', 'PXA', 'PXI', 'PXL', 'PXM', 'PXS', 'PXT'
]
# ============================================
# PHẦN 2D: DATABASE MÃ CỔ PHIẾU Q-T (Nhóm 4/5)
# ============================================

STOCKS_Q_TO_T = [
    'QBS', 'QCC', 'QCG', 'QHD', 'QHW', 'QLT', 'QNC', 'QNS', 'QNW', 'QNU', 'QST', 'QTC',
    'RAL', 'RAT', 'RBC', 'RCC', 'RCD', 'RCL', 'RDP', 'RED', 'REE', 'RGC', 'RHC', 'RHN', 'RIC',
    'ROG', 'ROM', 'ROS', 'RTB', 'RTC', 'RTT', 'RUB', 'RUM',
    'S12', 'S27', 'S33', 'S4A', 'S55', 'S64', 'S72', 'S74', 'S91', 'S96', 'S99', 'SAB', 'SAC',
    'SAF', 'SAL', 'SAM', 'SAP', 'SAS', 'SAV', 'SBA', 'SBB', 'SBC', 'SBD', 'SBH', 'SBL', 'SBM',
    'SBR', 'SBS', 'SBT', 'SBV', 'SC5', 'SCA', 'SCD', 'SCG', 'SCI', 'SCJ', 'SCL', 'SCO', 'SCR',
    'SCS', 'SCY', 'SEB', 'SED', 'SEP', 'SFC', 'SFG', 'SFI', 'SFN', 'SGB', 'SGC', 'SGD', 'SGH',
    'SGI', 'SGN', 'SGO', 'SGP', 'SGR', 'SGT', 'SGU', 'SHA', 'SHB', 'SHE', 'SHG', 'SHI', 'SHN',
    'SHP', 'SHS', 'SHV', 'SII', 'SIP', 'SJ1', 'SJC', 'SJD', 'SJE', 'SJF', 'SJG', 'SJM', 'SKG',
    'SKH', 'SKN', 'SKS', 'SKV', 'SLC', 'SLS', 'SMA', 'SMB', 'SMC', 'SMN', 'SMS', 'SMT', 'SNC',
    'SNG', 'SNZ', 'SOS', 'SPB', 'SPC', 'SPD', 'SPM', 'SPN', 'SPP', 'SPS', 'SQC', 'SRA', 'SRC',
    'SRF', 'SSB', 'SSC', 'SSF', 'SSG', 'SSH', 'SSI', 'SSM', 'SSN', 'SSS', 'ST8', 'STC', 'STD',
    'STG', 'STH', 'STK', 'STL', 'STN', 'STP', 'STR', 'STS', 'STT', 'STV', 'STW', 'SVC', 'SVD',
    'SVG', 'SVH', 'SVI', 'SVN', 'SVS', 'SVT', 'SZB', 'SZC', 'SZE', 'SZG', 'SZL',
    'TA3', 'TA6', 'TA9', 'TAC', 'TAN', 'TAR', 'TAS', 'TB8', 'TBC', 'TBD', 'TBH', 'TBT', 'TBX',
    'TC6', 'TCB', 'TCD', 'TCH', 'TCI', 'TCL', 'TCM', 'TCO', 'TCR', 'TCS', 'TCT', 'TCW', 'TDA',
    'TDB', 'TDC', 'TDF', 'TDG', 'TDH', 'TDM', 'TDN', 'TDP', 'TDS', 'TDT', 'TDW', 'TEG', 'TET',
    'TFC', 'TGG', 'TGP', 'THB', 'THD', 'THG', 'THI', 'THP', 'THR', 'THS', 'THT', 'THU', 'THV',
    'THW', 'TIC', 'TID', 'TIE', 'TIG', 'TIP', 'TIS', 'TIX', 'TJC', 'TKA', 'TKC', 'TKG', 'TKU',
    'TL4', 'TLB', 'TLC', 'TLD', 'TLG', 'TLH', 'TLI', 'TLP', 'TLT', 'TM6', 'TMA', 'TMB', 'TMC',
    'TMG', 'TMP', 'TMS', 'TMT', 'TMW', 'TMX', 'TN1', 'TNA', 'TNC', 'TNG', 'TNH', 'TNI', 'TNM',
    'TNP', 'TNS', 'TNT', 'TNW', 'TOP', 'TOT', 'TOW', 'TPC', 'TPB', 'TPH', 'TPP', 'TPS', 'TQN',
    'TQW', 'TR1', 'TRA', 'TRB', 'TRC', 'TRI', 'TRS', 'TS3', 'TS4', 'TS5', 'TSB', 'TSC', 'TSD',
    'TSG', 'TSJ', 'TSM', 'TST', 'TTA', 'TTB', 'TTC', 'TTD', 'TTE', 'TTF', 'TTG', 'TTH', 'TTJ',
    'TTL', 'TTN', 'TTP', 'TTR', 'TTS', 'TTT', 'TTV', 'TTZ', 'TUG', 'TV1', 'TV2', 'TV3', 'TV4',
    'TV6', 'TVA', 'TVB', 'TVC', 'TVD', 'TVG', 'TVH', 'TVM', 'TVN', 'TVP', 'TVS', 'TVT', 'TVU',
    'TVW', 'TXM', 'TYA'
]
# ============================================
# PHẦN 2E: DATABASE MÃ CỔ PHIẾU U-Z (Nhóm 5/5) + TỔNG HỢP
# ============================================

STOCKS_U_TO_Z = [
    'UDC', 'UDJ', 'UDL', 'UEM', 'UIC', 'UMC', 'UNI', 'UPC', 'USC',
    'V11', 'V12', 'V15', 'V21', 'VAB', 'VAF', 'VAS', 'VAV', 'VBC', 'VBG', 'VBH', 'VC1', 'VC2',
    'VC3', 'VC5', 'VC6', 'VC7', 'VC9', 'VCA', 'VCB', 'VCC', 'VCE', 'VCF', 'VCG', 'VCH', 'VCI',
    'VCM', 'VCP', 'VCR', 'VCS', 'VCT', 'VCV', 'VCW', 'VCX', 'VDB', 'VDL', 'VDN', 'VDP', 'VDS',
    'VDT', 'VE1', 'VE2', 'VE3', 'VE4', 'VE8', 'VE9', 'VES', 'VET', 'VFC', 'VFG', 'VFR', 'VFS',
    'VGC', 'VGI', 'VGL', 'VGP', 'VGR', 'VGS', 'VGT', 'VGV', 'VHC', 'VHD', 'VHE', 'VHF', 'VHG',
    'VHH', 'VHI', 'VHL', 'VHM', 'VIB', 'VIC', 'VID', 'VIE', 'VIF', 'VIG', 'VIM', 'VIN', 'VIP',
    'VIR', 'VIS', 'VIT', 'VIW', 'VIX', 'VJC', 'VKC', 'VLA', 'VLB', 'VLC', 'VLF', 'VLG', 'VLW',
    'VMC', 'VMD', 'VMG', 'VMI', 'VMS', 'VNA', 'VNB', 'VNC', 'VND', 'VNE', 'VNF', 'VNG', 'VNH',
    'VNI', 'VNL', 'VNM', 'VNP', 'VNR', 'VNS', 'VNT', 'VNX', 'VOC', 'VOS', 'VPB', 'VPC', 'VPD',
    'VPG', 'VPH', 'VPI', 'VPR', 'VPS', 'VPW', 'VQC', 'VRC', 'VRE', 'VSA', 'VSC', 'VSE', 'VSF',
    'VSG', 'VSH', 'VSI', 'VSM', 'VSN', 'VSP', 'VST', 'VTA', 'VTB', 'VTC', 'VTE', 'VTF', 'VTG',
    'VTH', 'VTI', 'VTJ', 'VTK', 'VTL', 'VTO', 'VTP', 'VTQ', 'VTR', 'VTS', 'VTV', 'VTX', 'VTZ',
    'VUC', 'VW3', 'VWS', 'VXB', 'VXP',
    'WSB', 'WSS',
    'X18', 'X20', 'X26', 'X77', 'XDH', 'XHC', 'XLV', 'XMC', 'XMD', 'XMP', 'XPH',
    'YBC', 'YBM', 'YEG', 'YTC',
    'ZTA'
]

# ============================================
# TỔNG HỢP TẤT CẢ MÃ CỔ PHIẾU (1800+)
# ============================================
# Ghép tất cả các phần lại với nhau
# Import từ các file part2, part2b, part2c, part2d, part2e
# hoặc copy paste tất cả vào một list

ALL_VN_STOCKS = (
    STOCKS_A_TO_D +
    STOCKS_E_TO_L +
    STOCKS_M_TO_P +
    STOCKS_Q_TO_T +
    STOCKS_U_TO_Z
)

# Loại bỏ duplicate và sort
ALL_VN_STOCKS = sorted(list(set(ALL_VN_STOCKS)))

print(f"✅ Tổng số mã cổ phiếu: {len(ALL_VN_STOCKS)}")
# ============================================
# PHẦN 2F: PHÂN LOẠI CỔ PHIẾU THEO NGÀNH
# ============================================

VN_STOCKS_BY_SECTOR = {
    'Ngân hàng': ['ACB', 'BAB', 'BID', 'BVB', 'CTG', 'EIB', 'HDB', 'KLB', 'LPB', 'MBB', 'MSB',
                  'NAB', 'NCB', 'NVB', 'OCB', 'PGB', 'SCB', 'SGB', 'SHB', 'SSB', 'STB', 'TCB',
                  'TPB', 'VAB', 'VBB', 'VCB', 'VIB', 'VPB'],

    'Chứng khoán': ['AGR', 'APS', 'ART', 'BSC', 'BSI', 'BVS', 'CTS', 'EVS', 'FTS', 'HCM', 'IVS',
                    'MBS', 'ORS', 'PSI', 'SHS', 'SSI', 'TVB', 'VCI', 'VDS', 'VIG', 'VIX', 'VND'],

    'Bất động sản': ['ASM', 'BCI', 'BCM', 'CEO', 'CIG', 'CII', 'DIG', 'DRH', 'DXG', 'DXS', 'FLC',
                     'HAG', 'HDC', 'HDG', 'HQC', 'IDC', 'ITA', 'KBC', 'KDH', 'LDG', 'LHG', 'NLG',
                     'NTL', 'NVL', 'PDR', 'PPI', 'QCG', 'SCR', 'SIP', 'SJS', 'SZC', 'TDC', 'TDH',
                     'VHM', 'VIC', 'VPI', 'VRE'],

    'Xây dựng': ['C4G', 'CC1', 'CII', 'CTD', 'CTI', 'CVT', 'DPG', 'FCN', 'HBC', 'HT1', 'HTN',
                 'LCG', 'PC1', 'PCC', 'PXI', 'REE', 'SC5', 'SCG', 'SZL', 'TCO', 'THG', 'VC3',
                 'VCG', 'VE1', 'VE3', 'VE4', 'VE8', 'VE9'],

    'Thép & Kim loại': ['DTL', 'DXV', 'GVR', 'HMC', 'HPG', 'HSG', 'KSB', 'NKG', 'POM', 'SMC',
                        'TLH', 'TVN', 'VGS'],

    'Dầu khí': ['ASP', 'BSR', 'CNG', 'DVP', 'GAS', 'HFC', 'OIL', 'PGC', 'PGD', 'PGI', 'PGS',
                'PLC', 'PLX', 'POS', 'POW', 'PSH', 'PVB', 'PVC', 'PVD', 'PVG', 'PVS', 'PVT',
                'PXS', 'PXT'],

    'Điện lực & Năng lượng': ['GEG', 'GEX', 'HND', 'NT2', 'POW', 'QTP', 'REE', 'SBA', 'TBC', 'VSH'],

    'Bán lẻ': ['ABA', 'ABT', 'AST', 'BBC', 'DGW', 'FRT', 'MWG', 'PAN', 'PET', 'PNJ', 'SAM',
               'SFI', 'VGC', 'VHC'],

    'Thực phẩm & Đồ uống': ['ABT', 'ACL', 'AGF', 'BAF', 'BBC', 'BHS', 'CAN', 'HNG', 'KDC', 'LAF',
                            'MCH', 'MML', 'MSN', 'NHS', 'ORN', 'QNS', 'SAB', 'SAV', 'SBT', 'SGT',
                            'TAC', 'TLG', 'TS4', 'VHC', 'VIF', 'VNM', 'VSN'],

    'Dược phẩm & Y tế': ['ADP', 'AGP', 'AMV', 'DBD', 'DCL', 'DHG', 'DHT', 'DMC', 'DP1', 'DP2',
                         'DP3', 'DVN', 'IMP', 'PME', 'PPP', 'TRA', 'VMD'],

    'Công nghệ': ['BMI', 'CMG', 'CMT', 'CMX', 'CNT', 'CTR', 'DAG', 'DGT', 'ELC', 'FPT', 'ICT',
                  'ITD', 'MFS', 'SAM', 'SGD', 'SGN', 'SGR', 'ST8', 'SVT', 'TDG', 'VGI', 'VNR',
                  'VNT'],

    'Vận tải & Logistics': ['ACV', 'ATA', 'CAV', 'CLW', 'GMD', 'GSP', 'HAH', 'HTV', 'HVN', 'IDV',
                            'PAN', 'PJT', 'PVT', 'SCS', 'STG', 'TCL', 'TMS', 'VFC', 'VJC', 'VOS',
                            'VSC', 'VTO'],

    'Vật liệu xây dựng': ['BCC', 'BMP', 'BTS', 'C32', 'DHA', 'DPR', 'DCM', 'HOM', 'HT1', 'KSB',
                          'NNC', 'PAN', 'PC1', 'SCG', 'TLH', 'VCM', 'VCS', 'VGC'],

    'Hóa chất': ['AAA', 'BFC', 'BTC', 'CSV', 'DAG', 'DGC', 'DPM', 'DRC', 'GVR', 'LAS', 'NCS',
                 'PAC', 'PLC', 'PMB', 'PTB', 'SFG', 'TNC', 'VFG'],

    'Cao su & Nhựa': ['BRC', 'CSM', 'DPR', 'DRC', 'GVR', 'HRC', 'PHR', 'TNC', 'TRC', 'VHG'],

    'Thủy sản': ['AAM', 'ABT', 'ACL', 'AGF', 'ANV', 'BLF', 'CMX', 'FMC', 'IDI', 'MPC', 'SJ1',
                 'TS4', 'VHC'],

    'Điện tử': ['CMG', 'DGW', 'FPT', 'ITD', 'SAM', 'ST8'],

    'Du lịch & Giải trí': ['CDO', 'DAH', 'DLG', 'HOT', 'OCH', 'PDN', 'PGT', 'PNG', 'SHN', 'TCH',
                           'VNG'],

    'Dệt may': ['ACL', 'AGM', 'GIL', 'HMC', 'MSH', 'NPS', 'PHT', 'STK', 'TNG', 'VGT'],

    'Giấy': ['AAA', 'BMP', 'DHC', 'GDT', 'MCV', 'SFC', 'TPC', 'VPG'],

    'Khoáng sản': ['BMW', 'BXH', 'CLC', 'DHM', 'DIC', 'DQC', 'KSH', 'MBG', 'NBC', 'PLC', 'THT',
                   'TMX']
}

# Tạo mapping ngược: từ mã cổ phiếu -> ngành
STOCK_TO_SECTOR = {}
for sector, stocks in VN_STOCKS_BY_SECTOR.items():
    for stock in stocks:
        STOCK_TO_SECTOR[stock] = sector

print(f"✅ Số ngành: {len(VN_STOCKS_BY_SECTOR)}")
print(f"✅ Đã mapping {len(STOCK_TO_SECTOR)} mã cổ phiếu vào ngành")
# ============================================
# PHẦN 3: SESSION STATE & VISITOR TRACKING (FIXED)
# ============================================

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
import time

# ============================================
# 3A: CẤU HÌNH FILE LƯU TRỮ
# ============================================

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

VISITOR_FILE = DATA_DIR / "visitors.json"
STATS_FILE = DATA_DIR / "stats.json"
ONLINE_FILE = DATA_DIR / "online_users.json"  # MỚI: Track online users


# ============================================
# 3B: HÀM QUẢN LÝ ONLINE USERS
# ============================================

def load_online_users():
    """Load danh sách users đang online"""
    try:
        if ONLINE_FILE.exists():
            with open(ONLINE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Xóa users không active trong 5 phút
            current_time = time.time()
            active_users = {
                sid: timestamp
                for sid, timestamp in data.items()
                if current_time - timestamp < 300  # 5 phút
            }

            # Lưu lại danh sách đã clean
            save_online_users(active_users)
            return active_users
        return {}
    except Exception as e:
        print(f"Error loading online users: {e}")
        return {}


def save_online_users(users_dict):
    """Lưu danh sách users online"""
    try:
        with open(ONLINE_FILE, 'w', encoding='utf-8') as f:
            json.dump(users_dict, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving online users: {e}")


def update_user_activity(session_id):
    """Cập nhật hoạt động của user"""
    online_users = load_online_users()
    online_users[session_id] = time.time()
    save_online_users(online_users)
    return len(online_users)


# ============================================
# 3C: HÀM QUẢN LÝ VISITOR DATA
# ============================================

def load_visitor_data():
    """Load dữ liệu visitor từ file"""
    try:
        if VISITOR_FILE.exists():
            with open(VISITOR_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            'total_visits': 0,
            'unique_visitors': [],
            'visit_history': []
        }
    except Exception as e:
        print(f"Error loading visitor data: {e}")
        return {
            'total_visits': 0,
            'unique_visitors': [],
            'visit_history': []
        }


def save_visitor_data(data):
    """Lưu dữ liệu visitor vào file"""
    try:
        with open(VISITOR_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving visitor data: {e}")


def load_stats():
    """Load thống kê từ file"""
    try:
        if STATS_FILE.exists():
            with open(STATS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            'daily_visits': {},
            'peak_online': 0,
            'peak_online_time': None,
            'total_searches': 0,
            'popular_stocks': {}
        }
    except Exception as e:
        print(f"Error loading stats: {e}")
        return {
            'daily_visits': {},
            'peak_online': 0,
            'peak_online_time': None,
            'total_searches': 0,
            'popular_stocks': {}
        }


def save_stats(stats):
    """Lưu thống kê vào file"""
    try:
        with open(STATS_FILE, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving stats: {e}")


# ============================================
# 3D: SESSION STATE INITIALIZATION
# ============================================

def initialize_session_state():
    """Khởi tạo session state với tracking"""

    # Tạo session ID duy nhất
    if 'session_id' not in st.session_state:
        timestamp = str(datetime.now().timestamp())
        random_str = str(time.time())
        st.session_state.session_id = hashlib.md5(f"{timestamp}-{random_str}".encode()).hexdigest()
        st.session_state.is_new_visitor = True
        st.session_state.last_activity = time.time()
    else:
        st.session_state.is_new_visitor = False

    # Cập nhật activity mỗi lần refresh
    current_online = update_user_activity(st.session_state.session_id)
    st.session_state.online_count = current_online

    # Load visitor data
    if 'visitor_data' not in st.session_state:
        st.session_state.visitor_data = load_visitor_data()

    # Load stats
    if 'stats' not in st.session_state:
        st.session_state.stats = load_stats()

    # Cập nhật total visits cho new visitor
    if st.session_state.is_new_visitor:
        visitor_data = st.session_state.visitor_data

        # Tăng total visits
        visitor_data['total_visits'] += 1

        # Thêm vào unique visitors
        if st.session_state.session_id not in visitor_data['unique_visitors']:
            visitor_data['unique_visitors'].append(st.session_state.session_id)

        # Thêm vào visit history
        visitor_data['visit_history'].append({
            'session_id': st.session_state.session_id,
            'timestamp': datetime.now().isoformat(),
            'date': datetime.now().strftime('%Y-%m-%d')
        })

        # Giới hạn history
        if len(visitor_data['visit_history']) > 1000:
            visitor_data['visit_history'] = visitor_data['visit_history'][-1000:]

        # Cập nhật daily visits
        today = datetime.now().strftime('%Y-%m-%d')
        stats = st.session_state.stats
        if today not in stats['daily_visits']:
            stats['daily_visits'][today] = 0
        stats['daily_visits'][today] += 1

        # Lưu lại
        st.session_state.visitor_data = visitor_data
        st.session_state.stats = stats
        save_visitor_data(visitor_data)
        save_stats(stats)

        # Đánh dấu không còn new
        st.session_state.is_new_visitor = False

    # Cập nhật peak online
    if current_online > st.session_state.stats.get('peak_online', 0):
        st.session_state.stats['peak_online'] = current_online
        st.session_state.stats['peak_online_time'] = datetime.now().isoformat()
        save_stats(st.session_state.stats)

    # Session state cho app features
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []

    if 'favorite_stocks' not in st.session_state:
        st.session_state.favorite_stocks = []

    if 'last_update' not in st.session_state:
        st.session_state.last_update = None


# ============================================
# 3E: HÀM TRACKING ACTIONS
# ============================================

def track_stock_search(symbol):
    """Track khi user search một mã cổ phiếu"""
    # Thêm vào search history của session
    if symbol not in st.session_state.search_history:
        st.session_state.search_history.append(symbol)

    # Cập nhật popular stocks
    stats = st.session_state.stats
    if 'popular_stocks' not in stats:
        stats['popular_stocks'] = {}

    if symbol not in stats['popular_stocks']:
        stats['popular_stocks'][symbol] = 0
    stats['popular_stocks'][symbol] += 1

    # Tăng total searches
    if 'total_searches' not in stats:
        stats['total_searches'] = 0
    stats['total_searches'] += 1

    st.session_state.stats = stats
    save_stats(stats)


def get_popular_stocks(top_n=10):
    """Lấy top mã cổ phiếu được search nhiều nhất"""
    stats = st.session_state.stats
    if 'popular_stocks' not in stats or not stats['popular_stocks']:
        return []

    sorted_stocks = sorted(
        stats['popular_stocks'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    return sorted_stocks[:top_n]


def get_visitor_stats():
    """Lấy thống kê visitor"""
    visitor_data = st.session_state.visitor_data
    stats = st.session_state.stats

    # Tính unique visitors hôm nay
    today = datetime.now().strftime('%Y-%m-%d')
    today_visitors = sum(
        1 for visit in visitor_data.get('visit_history', [])
        if visit.get('date') == today
    )

    # Lấy số online từ file (real-time)
    online_users = load_online_users()
    online_count = len(online_users)

    return {
        'total_visits': visitor_data.get('total_visits', 0),
        'unique_visitors': len(visitor_data.get('unique_visitors', [])),
        'online_now': online_count,
        'today_visits': stats.get('daily_visits', {}).get(today, 0),
        'peak_online': stats.get('peak_online', 0),
        'total_searches': stats.get('total_searches', 0)
    }


# ============================================
# 3F: AUTO CLEANUP
# ============================================

def cleanup_old_data():
    """Xóa dữ liệu cũ hơn 30 ngày"""
    try:
        visitor_data = st.session_state.visitor_data
        cutoff_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

        # Lọc visit history
        visitor_data['visit_history'] = [
            visit for visit in visitor_data.get('visit_history', [])
            if visit.get('date', '9999-99-99') >= cutoff_date
        ]

        # Lọc daily visits
        stats = st.session_state.stats
        stats['daily_visits'] = {
            date: count for date, count in stats.get('daily_visits', {}).items()
            if date >= cutoff_date
        }

        save_visitor_data(visitor_data)
        save_stats(stats)

    except Exception as e:
        print(f"Error cleaning up: {e}")


# ============================================
# 3G: BACKGROUND HEARTBEAT (Cập nhật liên tục)
# ============================================

def keep_alive():
    """Cập nhật trạng thái online mỗi lần user tương tác"""
    current_time = time.time()

    # Chỉ update nếu đã qua 30 giây kể từ lần cuối
    if current_time - st.session_state.get('last_activity', 0) > 30:
        update_user_activity(st.session_state.session_id)
        st.session_state.last_activity = current_time


# ============================================
# 3H: KHỞI TẠO
# ============================================

# Gọi khi app start
initialize_session_state()

# Cleanup định kỳ
import random

if random.random() < 0.01:
    cleanup_old_data()

# Keep alive
keep_alive()

print("✅ Session State & Visitor Tracking initialized (FIXED)")
print(f"   - Session ID: {st.session_state.session_id[:8]}...")
print(f"   - Total visits: {st.session_state.visitor_data.get('total_visits', 0)}")
print(f"   - Online now: {st.session_state.online_count}")
# ============================================
# PHẦN 4 ULTIMATE: 60+ CHỈ BÁO KỸ THUẬT
# ============================================

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# ============================================
# 4A: HÀM LẤY DỮ LIỆU
# ============================================

@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_data(symbol, period='1y'):
    """Lấy dữ liệu cổ phiếu từ Yahoo Finance"""
    try:
        ticker = yf.Ticker(f"{symbol}.VN")
        df = ticker.history(period=period)
        info = ticker.info

        if df.empty:
            return None, None

        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })

        return df, info
    except Exception as e:
        print(f"Error getting data for {symbol}: {e}")
        return None, None


# ============================================
# 4B: 60+ CHỈ BÁO KỸ THUẬT
# ============================================

def calculate_advanced_indicators(df):
    """Tính toán 60+ chỉ báo kỹ thuật"""
    if df is None or df.empty:
        return df

    # ========== NHÓM 1: MOVING AVERAGES (9 chỉ báo) ==========
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA10'] = df['close'].rolling(window=10).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()
    df['MA100'] = df['close'].rolling(window=100).mean()
    df['MA200'] = df['close'].rolling(window=200).mean()
    df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()

    # ========== NHÓM 2: ADVANCED MAs (5 chỉ báo) ========== MỚI!
    # Hull Moving Average
    wma_half = df['close'].rolling(window=10).apply(lambda x: np.dot(x, np.arange(1, 11)) / np.arange(1, 11).sum(),
                                                    raw=True)
    wma_full = df['close'].rolling(window=20).apply(lambda x: np.dot(x, np.arange(1, 21)) / np.arange(1, 21).sum(),
                                                    raw=True)
    df['HMA'] = (2 * wma_half - wma_full).rolling(window=5).apply(
        lambda x: np.dot(x, np.arange(1, 6)) / np.arange(1, 6).sum(), raw=True
    )

    # DEMA - Double Exponential MA
    ema1 = df['close'].ewm(span=20, adjust=False).mean()
    ema2 = ema1.ewm(span=20, adjust=False).mean()
    df['DEMA'] = 2 * ema1 - ema2

    # TEMA - Triple Exponential MA
    ema3 = ema2.ewm(span=20, adjust=False).mean()
    df['TEMA'] = 3 * ema1 - 3 * ema2 + ema3

    # VWMA - Volume Weighted MA
    df['VWMA'] = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()

    # ZLEMA - Zero Lag EMA
    lag = (20 - 1) // 2
    df['ZLEMA'] = (df['close'] + (df['close'] - df['close'].shift(lag))).ewm(span=20, adjust=False).mean()

    # ========== NHÓM 3: RSI & MOMENTUM (8 chỉ báo) ==========
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Stochastic RSI - MỚI!
    rsi_low = df['RSI'].rolling(window=14).min()
    rsi_high = df['RSI'].rolling(window=14).max()
    df['StochRSI'] = 100 * (df['RSI'] - rsi_low) / (rsi_high - rsi_low)
    df['StochRSI_K'] = df['StochRSI'].rolling(window=3).mean()
    df['StochRSI_D'] = df['StochRSI_K'].rolling(window=3).mean()

    # TSI - True Strength Index - MỚI!
    momentum = df['close'].diff()
    double_smoothed_mom = momentum.ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
    double_smoothed_abs_mom = momentum.abs().ewm(span=25, adjust=False).mean().ewm(span=13, adjust=False).mean()
    df['TSI'] = 100 * (double_smoothed_mom / double_smoothed_abs_mom)

    # CMO - Chande Momentum Oscillator - MỚI!
    mom_sum = momentum.rolling(window=14).sum()
    abs_mom_sum = momentum.abs().rolling(window=14).sum()
    df['CMO'] = 100 * (mom_sum / abs_mom_sum)

    # ========== NHÓM 4: MACD (3 chỉ báo) ==========
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    # ========== NHÓM 5: BOLLINGER & KELTNER (8 chỉ báo) ==========
    df['BB_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']

    # ATR calculation
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()
    df['ATR'] = atr

    df['Keltner_middle'] = df['close'].ewm(span=20, adjust=False).mean()
    df['Keltner_upper'] = df['Keltner_middle'] + (2 * atr)
    df['Keltner_lower'] = df['Keltner_middle'] - (2 * atr)

    # ========== NHÓM 6: DONCHIAN CHANNEL (3 chỉ báo) ========== MỚI!
    df['Donchian_upper'] = df['high'].rolling(window=20).max()
    df['Donchian_lower'] = df['low'].rolling(window=20).min()
    df['Donchian_middle'] = (df['Donchian_upper'] + df['Donchian_lower']) / 2

    # ========== NHÓM 7: STOCHASTIC (2 chỉ báo) ==========
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['Stoch_K'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()

    # ========== NHÓM 8: WILLIAMS %R (1 chỉ báo) ==========
    df['Williams_R'] = -100 * ((high_14 - df['close']) / (high_14 - low_14))

    # ========== NHÓM 9: ADX & DIRECTIONAL (3 chỉ báo) ==========
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    df['ADX'] = dx.rolling(window=14).mean()
    df['DI_plus'] = plus_di
    df['DI_minus'] = minus_di

    # ========== NHÓM 10: SUPERTREND (2 chỉ báo) ========== MỚI!
    multiplier = 3
    hl_avg = (df['high'] + df['low']) / 2
    basic_ub = hl_avg + (multiplier * atr)
    basic_lb = hl_avg - (multiplier * atr)

    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    for i in range(len(df)):
        if i == 0:
            supertrend.iloc[i] = basic_ub.iloc[i]
            direction.iloc[i] = 1
        else:
            if df['close'].iloc[i] > supertrend.iloc[i - 1]:
                supertrend.iloc[i] = max(basic_lb.iloc[i], supertrend.iloc[i - 1])
                direction.iloc[i] = 1
            elif df['close'].iloc[i] < supertrend.iloc[i - 1]:
                supertrend.iloc[i] = min(basic_ub.iloc[i], supertrend.iloc[i - 1])
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = supertrend.iloc[i - 1]
                direction.iloc[i] = direction.iloc[i - 1]

    df['Supertrend'] = supertrend
    df['Supertrend_direction'] = direction

    # ========== NHÓM 11: OBV & VOLUME (6 chỉ báo) ==========
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['Volume_MA'] = df['volume'].rolling(window=20).mean()
    df['Volume_ratio'] = df['volume'] / df['Volume_MA']

    # Force Index - MỚI!
    df['Force_Index'] = df['close'].diff() * df['volume']
    df['Force_Index_MA'] = df['Force_Index'].ewm(span=13, adjust=False).mean()

    # Ease of Movement - MỚI!
    distance = ((df['high'] + df['low']) / 2 - (df['high'].shift() + df['low'].shift()) / 2)
    box_ratio = (df['volume'] / 1000000) / (df['high'] - df['low'])
    df['EMV'] = distance / box_ratio
    df['EMV_MA'] = df['EMV'].rolling(window=14).mean()

    # ========== NHÓM 12: CCI & MFI (2 chỉ báo) ==========
    tp = (df['high'] + df['low'] + df['close']) / 3
    df['CCI'] = (tp - tp.rolling(window=20).mean()) / (0.015 * tp.rolling(window=20).std())

    typical_price = (df['high'] + df['low'] + df['close']) / 3
    raw_money_flow = typical_price * df['volume']
    positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=14).sum()
    negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=14).sum()
    mfi_ratio = positive_flow / negative_flow
    df['MFI'] = 100 - (100 / (1 + mfi_ratio))

    # ========== NHÓM 13: ROC (1 chỉ báo) ==========
    df['ROC'] = ((df['close'] - df['close'].shift(12)) / df['close'].shift(12)) * 100

    # ========== NHÓM 14: ICHIMOKU (4 chỉ báo) ==========
    high_9 = df['high'].rolling(window=9).max()
    low_9 = df['low'].rolling(window=9).min()
    df['Ichimoku_conversion'] = (high_9 + low_9) / 2

    high_26 = df['high'].rolling(window=26).max()
    low_26 = df['low'].rolling(window=26).min()
    df['Ichimoku_base'] = (high_26 + low_26) / 2

    df['Ichimoku_span_a'] = ((df['Ichimoku_conversion'] + df['Ichimoku_base']) / 2).shift(26)

    high_52 = df['high'].rolling(window=52).max()
    low_52 = df['low'].rolling(window=52).min()
    df['Ichimoku_span_b'] = ((high_52 + low_52) / 2).shift(26)

    # ========== NHÓM 15: VWAP & PIVOT (4 chỉ báo) ==========
    df['VWAP'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    df['Pivot'] = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
    df['Resistance1'] = 2 * df['Pivot'] - df['low'].shift(1)
    df['Support1'] = 2 * df['Pivot'] - df['high'].shift(1)

    # ========== NHÓM 16: ULTIMATE OSCILLATOR (1 chỉ báo) ==========
    bp = df['close'] - pd.concat([df['low'], df['close'].shift()], axis=1).min(axis=1)
    bp_sum_7 = bp.rolling(window=7).sum()
    bp_sum_14 = bp.rolling(window=14).sum()
    bp_sum_28 = bp.rolling(window=28).sum()

    tr_sum_7 = tr.rolling(window=7).sum()
    tr_sum_14 = tr.rolling(window=14).sum()
    tr_sum_28 = tr.rolling(window=28).sum()

    avg_7 = bp_sum_7 / tr_sum_7
    avg_14 = bp_sum_14 / tr_sum_14
    avg_28 = bp_sum_28 / tr_sum_28

    df['UltimateOsc'] = 100 * ((4 * avg_7) + (2 * avg_14) + avg_28) / 7

    # ========== NHÓM 17: CHAIKIN (1 chỉ báo) ==========
    adl = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']) * df['volume']
    adl = adl.fillna(0).cumsum()
    df['Chaikin'] = adl.ewm(span=3, adjust=False).mean() - adl.ewm(span=10, adjust=False).mean()

    # ========== NHÓM 18: AROON (2 chỉ báo) ==========
    aroon_period = 25
    df['Aroon_up'] = df['high'].rolling(window=aroon_period + 1).apply(
        lambda x: (aroon_period - x[::-1].argmax()) / aroon_period * 100, raw=True
    )
    df['Aroon_down'] = df['low'].rolling(window=aroon_period + 1).apply(
        lambda x: (aroon_period - x[::-1].argmin()) / aroon_period * 100, raw=True
    )

    # ========== NHÓM 19: VORTEX (2 chỉ báo) ========== MỚI!
    vi_plus = abs(df['high'] - df['low'].shift(1)).rolling(window=14).sum()
    vi_minus = abs(df['low'] - df['high'].shift(1)).rolling(window=14).sum()
    tr_sum = tr.rolling(window=14).sum()
    df['Vortex_plus'] = vi_plus / tr_sum
    df['Vortex_minus'] = vi_minus / tr_sum

    # ========== NHÓM 20: TRIX (1 chỉ báo) ==========
    ema1 = df['close'].ewm(span=15, adjust=False).mean()
    ema2 = ema1.ewm(span=15, adjust=False).mean()
    ema3 = ema2.ewm(span=15, adjust=False).mean()
    df['TRIX'] = (ema3 - ema3.shift(1)) / ema3.shift(1) * 100

    # ========== NHÓM 21: VOLATILITY (3 chỉ báo) ==========
    df['Volatility'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()
    df['True_Range'] = tr

    # Choppiness Index - MỚI!
    atr_sum = tr.rolling(window=14).sum()
    high_low_range = df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min()
    df['Choppiness'] = 100 * np.log10(atr_sum / high_low_range) / np.log10(14)

    # ========== NHÓM 22: ELDER RAY (2 chỉ báo) ========== MỚI!
    ema13 = df['close'].ewm(span=13, adjust=False).mean()
    df['Bull_Power'] = df['high'] - ema13
    df['Bear_Power'] = df['low'] - ema13

    # ========== NHÓM 23: AWESOME OSCILLATOR (1 chỉ báo) ========== MỚI!
    median_price = (df['high'] + df['low']) / 2
    df['AO'] = median_price.rolling(window=5).mean() - median_price.rolling(window=34).mean()

    # ========== NHÓM 24: CANDLESTICK PATTERNS (5 patterns) ========== MỚI!
    # Doji
    body = abs(df['close'] - df['open'])
    range_hl = df['high'] - df['low']
    df['Doji'] = (body / range_hl < 0.1).astype(int)

    # Hammer
    lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
    upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
    df['Hammer'] = ((lower_shadow > 2 * body) & (upper_shadow < body)).astype(int)

    # Shooting Star
    df['Shooting_Star'] = ((upper_shadow > 2 * body) & (lower_shadow < body)).astype(int)

    # Bullish Engulfing
    prev_body = abs(df['close'].shift(1) - df['open'].shift(1))
    df['Bullish_Engulfing'] = (
            (df['close'] > df['open']) &
            (df['close'].shift(1) < df['open'].shift(1)) &
            (df['close'] > df['open'].shift(1)) &
            (df['open'] < df['close'].shift(1)) &
            (body > prev_body)
    ).astype(int)

    # Bearish Engulfing
    df['Bearish_Engulfing'] = (
            (df['close'] < df['open']) &
            (df['close'].shift(1) > df['open'].shift(1)) &
            (df['close'] < df['open'].shift(1)) &
            (df['open'] > df['close'].shift(1)) &
            (body > prev_body)
    ).astype(int)

    return df


# ============================================
# 4C: DỰ ĐOÁN CẢI TIẾN
# ============================================

def predict_future_price_enhanced(df, days=7):
    """Dự đoán giá với nhiều phương pháp kết hợp"""
    if df is None or df.empty or len(df) < 30:
        return None, None

    try:
        recent_data = df['close'].tail(60).values

        # METHOD 1: Polynomial (40%)
        x = np.arange(len(recent_data))
        z = np.polyfit(x, recent_data, 3)
        p = np.poly1d(z)
        future_x = np.arange(len(recent_data), len(recent_data) + days)
        poly_pred = p(future_x)

        # METHOD 2: WMA (30%)
        weights = np.exp(np.linspace(-1, 0, 30))
        weights /= weights.sum()
        wma_value = np.dot(recent_data[-30:], weights)
        trend = (recent_data[-1] - recent_data[-30]) / 30
        wma_pred = wma_value + trend * np.arange(1, days + 1)

        # METHOD 3: Momentum adjustment (30%)
        latest = df.iloc[-1]

        rsi = latest['RSI']
        rsi_factor = 1.02 if rsi < 30 else 0.98 if rsi > 70 else 1.0

        macd_strength = latest['MACD'] - latest['MACD_signal']
        macd_factor = 1 + (macd_strength / abs(recent_data[-1])) * 0.5
        macd_factor = np.clip(macd_factor, 0.95, 1.05)

        # Supertrend adjustment
        if pd.notna(latest.get('Supertrend_direction')):
            supertrend_factor = 1.01 if latest['Supertrend_direction'] == 1 else 0.99
        else:
            supertrend_factor = 1.0

        momentum_factor = rsi_factor * macd_factor * supertrend_factor

        # Kết hợp
        combined_pred = (
                poly_pred * 0.4 +
                wma_pred * 0.3 +
                recent_data[-1] * momentum_factor * 0.3
        )

        # Volatility
        volatility = df['ATR'].iloc[-1] if pd.notna(df['ATR'].iloc[-1]) else df['close'].tail(30).std()
        noise = np.random.normal(0, volatility * 0.15, days)
        combined_pred = combined_pred + noise

        # Giới hạn
        combined_pred = np.maximum(combined_pred, recent_data[-1] * 0.7)
        combined_pred = np.minimum(combined_pred, recent_data[-1] * 1.3)

        # Confidence
        adx = latest['ADX'] if pd.notna(latest['ADX']) else 20
        vol_ratio = df['Volatility'].iloc[-1] if pd.notna(df['Volatility'].iloc[-1]) else 0.05

        confidence = (adx / 50) * 0.5 + (1 - min(vol_ratio * 10, 1)) * 0.5
        confidence = np.clip(confidence, 0.3, 0.9)

        return combined_pred, confidence

    except Exception as e:
        print(f"Error predicting: {e}")
        return None, None


# ============================================
# 4D: CÁC HÀM HỖ TRỢ KHÁC
# ============================================

def calculate_money_flow(df):
    """Phân tích dòng tiền"""
    if df is None or df.empty:
        return None

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    mfi = latest.get('MFI', 50)
    volume_change = ((latest['volume'] - prev['volume']) / prev['volume'] * 100) if prev['volume'] > 0 else 0
    obv_change = latest['OBV'] - df['OBV'].iloc[-20] if len(df) >= 20 else 0

    if mfi > 70:
        flow_status = "Dòng tiền mạnh - Quá mua"
        flow_color = "#FFA726"
    elif mfi > 50:
        flow_status = "Dòng tiền tích cực"
        flow_color = "#66BB6A"
    elif mfi > 30:
        flow_status = "Dòng tiền trung lập"
        flow_color = "#FDD835"
    else:
        flow_status = "Dòng tiền yếu - Quá bán"
        flow_color = "#EF5350"

    return {
        'MFI': mfi,
        'status': flow_status,
        'color': flow_color,
        'volume_change': volume_change,
        'obv_change': obv_change,
        'obv_trend': 'Tăng' if obv_change > 0 else 'Giảm'
    }


def get_support_resistance(df):
    """Tìm hỗ trợ & kháng cự"""
    if df is None or len(df) < 20:
        return None

    latest = df.iloc[-1]
    recent = df.tail(252) if len(df) >= 252 else df

    resistance_levels = [
        latest['BB_upper'] if pd.notna(latest['BB_upper']) else None,
        latest['Donchian_upper'] if pd.notna(latest.get('Donchian_upper')) else None,
        latest['Resistance1'] if pd.notna(latest['Resistance1']) else None,
        recent['high'].quantile(0.95),
        recent['high'].max()
    ]

    support_levels = [
        latest['BB_lower'] if pd.notna(latest['BB_lower']) else None,
        latest['Donchian_lower'] if pd.notna(latest.get('Donchian_lower')) else None,
        latest['Support1'] if pd.notna(latest['Support1']) else None,
        recent['low'].quantile(0.05),
        recent['low'].min()
    ]

    return {
        'resistance': [r for r in resistance_levels if r is not None],
        'support': [s for s in support_levels if s is not None],
        'current': latest['close']
    }


def get_price_change(df, periods=[1, 7, 30]):
    """Thay đổi giá"""
    if df is None or len(df) < 2:
        return {}

    latest_price = df['close'].iloc[-1]
    changes = {}

    for period in periods:
        if len(df) > period:
            old_price = df['close'].iloc[-period - 1]
            change = ((latest_price - old_price) / old_price * 100) if old_price > 0 else 0
            changes[f'{period}d'] = change

    return changes


def detect_candlestick_patterns(df):
    """Phát hiện mẫu nến"""
    if df is None or len(df) < 2:
        return {}

    latest = df.iloc[-1]

    patterns = {
        'Doji': latest.get('Doji', 0) == 1,
        'Hammer': latest.get('Hammer', 0) == 1,
        'Shooting Star': latest.get('Shooting_Star', 0) == 1,
        'Bullish Engulfing': latest.get('Bullish_Engulfing', 0) == 1,
        'Bearish Engulfing': latest.get('Bearish_Engulfing', 0) == 1
    }

    return {k: v for k, v in patterns.items() if v}


print("✅ Ultimate indicators (60+) loaded!")
# ============================================
# PHẦN 5 SMART: LOGIC PHÂN TÍCH THÔNG MINH
# ============================================

from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd


# ============================================
# 5A: TẠO TÍN HIỆU VỚI LOGIC THÔNG MINH
# ============================================

def generate_advanced_signal(df, info=None):
    """
    Tạo tín hiệu với logic thông minh:
    1. Kiểm tra điều kiện VETO (loại ngay nếu quá xấu)
    2. Tính điểm có trọng số
    3. Xác nhận chéo giữa các chỉ báo
    """
    if df is None or df.empty or len(df) < 50:
        return "N/A", 50, "Không đủ dữ liệu", "N/A", {}

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    score = 50
    reasons = []
    details = {}
    veto_reasons = []  # Lý do VETO

    # ========================================
    # BƯỚC 1: KIỂM TRA ĐIỀU KIỆN VETO (Loại ngay)
    # ========================================

    veto_count = 0

    # VETO 1: Supertrend GIẢM mạnh
    if pd.notna(latest.get('Supertrend_direction')):
        if latest['Supertrend_direction'] == -1:
            veto_count += 1
            veto_reasons.append("🚫 Supertrend GIẢM")

    # VETO 2: RSI quá mua + giá giảm
    if pd.notna(latest['RSI']):
        if latest['RSI'] > 75 and latest['close'] < prev['close']:
            veto_count += 1
            veto_reasons.append("🚫 RSI quá mua + giá giảm")

    # VETO 3: MACD Death Cross gần đây
    if pd.notna(latest['MACD']) and pd.notna(latest['MACD_signal']):
        if latest['MACD'] < latest['MACD_signal'] and prev['MACD'] >= prev['MACD_signal']:
            veto_count += 1
            veto_reasons.append("🚫 MACD Death Cross mới")

    # VETO 4: Death Cross MA (MA5 < MA20 < MA50)
    if pd.notna(latest['MA5']) and pd.notna(latest['MA20']) and pd.notna(latest['MA50']):
        if latest['MA5'] < latest['MA20'] < latest['MA50']:
            veto_count += 1
            veto_reasons.append("🚫 Death Cross (MA5<MA20<MA50)")

    # VETO 5: ADX yếu + RSI giảm
    if pd.notna(latest['ADX']) and pd.notna(latest['RSI']):
        if latest['ADX'] < 15 and latest['RSI'] < prev['RSI']:
            veto_count += 0.5
            veto_reasons.append("⚠️ Xu hướng rất yếu")

    # VETO 6: Bearish Engulfing
    if latest.get('Bearish_Engulfing', 0) == 1:
        veto_count += 1
        veto_reasons.append("🚫 Bearish Engulfing")

    # ========================================
    # BƯỚC 2: XỬ LÝ VETO
    # ========================================

    if veto_count >= 3:
        # Có >= 3 tín hiệu VETO → Buộc phải BÁN/GIỮ
        return "BÁN", 25, "🚫 CẢNH BÁO NGHIÊM TRỌNG:\n" + "\n".join(veto_reasons), "Nên bán ngay", {'veto': veto_count}
    elif veto_count >= 2:
        # Có 2 VETO → Điểm tối đa chỉ 45 (GIỮ)
        max_score = 45
        reasons.append("⚠️ Có 2 tín hiệu cảnh báo nghiêm trọng")
        reasons.extend(veto_reasons)
    elif veto_count >= 1:
        # Có 1 VETO → Điểm tối đa 60 (MUA thận trọng)
        max_score = 60
        reasons.append("⚠️ Có 1 tín hiệu cảnh báo")
        reasons.extend(veto_reasons)
    else:
        max_score = 100

    # ========================================
    # BƯỚC 3: TÍNH ĐIỂM CHI TIẾT (nếu pass VETO)
    # ========================================

    # === 1. XU HƯỚNG (25 điểm) ===
    trend_score = 0

    if pd.notna(latest['MA5']) and pd.notna(latest['MA20']) and pd.notna(latest['MA50']):
        # Golden Cross
        if latest['MA5'] > latest['MA20'] > latest['MA50']:
            trend_score += 15
            reasons.append("✅✅ Golden Cross mạnh")
        elif latest['MA5'] > latest['MA20']:
            trend_score += 10
            reasons.append("📈 MA ngắn hạn tăng")
        elif latest['MA5'] < latest['MA20']:
            trend_score -= 8
            reasons.append("⚠️ MA ngắn hạn giảm")

        if latest['close'] > latest['MA50']:
            trend_score += 5
        if pd.notna(latest['MA200']) and latest['close'] > latest['MA200']:
            trend_score += 5

    score += trend_score
    details['trend_score'] = trend_score

    # === 2. SUPERTREND (20 điểm) ===
    supertrend_score = 0
    if pd.notna(latest.get('Supertrend_direction')):
        if latest['Supertrend_direction'] == 1:
            supertrend_score += 20
            reasons.append("✅✅✅ Supertrend TĂNG")
        else:
            supertrend_score -= 20
            # Đã xử lý ở VETO

    score += supertrend_score
    details['supertrend_score'] = supertrend_score

    # === 3. RSI (18 điểm) ===
    rsi_score = 0
    if pd.notna(latest['RSI']):
        rsi = latest['RSI']
        if 40 <= rsi <= 60:
            rsi_score += 15
            reasons.append(f"✅ RSI lành mạnh ({rsi:.1f})")
        elif 30 <= rsi < 40:
            rsi_score += 12
            reasons.append(f"💰 RSI hấp dẫn ({rsi:.1f})")
        elif rsi < 30:
            rsi_score += 10
            reasons.append(f"💰💰 RSI quá bán ({rsi:.1f})")
        elif 60 < rsi <= 70:
            rsi_score += 8
            reasons.append(f"📈 RSI tích cực ({rsi:.1f})")
        elif 70 < rsi <= 80:
            rsi_score -= 8
            reasons.append(f"⚠️ RSI cao ({rsi:.1f})")
        else:  # > 80
            rsi_score -= 15
            reasons.append(f"⚠️⚠️ RSI quá mua ({rsi:.1f})")

    score += rsi_score
    details['rsi_score'] = rsi_score

    # === 4. MACD (15 điểm) ===
    macd_score = 0
    if pd.notna(latest['MACD']) and pd.notna(latest['MACD_signal']):
        if latest['MACD'] > latest['MACD_signal'] and prev['MACD'] <= prev['MACD_signal']:
            macd_score += 15
            reasons.append("✅✅ MACD Golden Cross")
        elif latest['MACD'] > latest['MACD_signal']:
            macd_score += 10
            reasons.append("✅ MACD dương")
        elif latest['MACD'] < latest['MACD_signal']:
            macd_score -= 8
            # Death Cross đã xử lý ở VETO

    score += macd_score
    details['macd_score'] = macd_score

    # === 5. ADX (12 điểm) ===
    adx_score = 0
    if pd.notna(latest['ADX']):
        if latest['ADX'] > 30:
            adx_score += 12
            reasons.append(f"✅✅ Xu hướng rất mạnh (ADX: {latest['ADX']:.1f})")
        elif latest['ADX'] > 25:
            adx_score += 8
            reasons.append(f"✅ Xu hướng mạnh (ADX: {latest['ADX']:.1f})")
        elif latest['ADX'] < 15:
            adx_score -= 6
            reasons.append(f"⚠️ Xu hướng rất yếu (ADX: {latest['ADX']:.1f})")

    score += adx_score
    details['adx_score'] = adx_score

    # === 6. VOLUME (10 điểm) ===
    volume_score = 0
    if pd.notna(latest['Volume_ratio']):
        if latest['Volume_ratio'] > 2.5:
            volume_score += 10
            reasons.append("✅✅✅ Volume bùng nổ")
        elif latest['Volume_ratio'] > 1.8:
            volume_score += 7
            reasons.append("✅✅ Volume rất cao")
        elif latest['Volume_ratio'] > 1.2:
            volume_score += 4
            reasons.append("✅ Volume tốt")
        elif latest['Volume_ratio'] < 0.6:
            volume_score -= 5
            reasons.append("⚠️ Volume yếu")

    score += volume_score
    details['volume_score'] = volume_score

    # === 7. CANDLESTICK PATTERNS (Bonus) ===
    patterns = detect_candlestick_patterns(df)

    if patterns.get('Bullish Engulfing'):
        score += 8
        reasons.append("✅✅ Bullish Engulfing")
    elif patterns.get('Hammer'):
        score += 5
        reasons.append("✅ Hammer")

    # ========================================
    # BƯỚC 4: GIỚI HẠN THEO VETO
    # ========================================

    score = max(0, min(score, max_score))

    # ========================================
    # BƯỚC 5: XÁC ĐỊNH TÍN HIỆU CUỐI CÙNG
    # ========================================

    # Logic thông minh: Kiểm tra tính nhất quán
    negative_count = sum(1 for r in reasons if '⚠️' in r or '🚫' in r)
    positive_count = sum(1 for r in reasons if '✅' in r or '💰' in r)

    # Nếu có quá nhiều lý do tiêu cực so với tích cực
    if negative_count > positive_count and score > 50:
        score = max(score - 15, 40)  # Phạt điểm nặng
        reasons.insert(0, "⚠️ CẢNH BÁO: Nhiều tín hiệu tiêu cực")

    if score >= 80:
        signal = "MUA MẠNH"
        term = "Ngắn & Trung hạn"
    elif score >= 70:
        signal = "MUA"
        term = "Ngắn hạn"
    elif score >= 60:
        signal = "MUA (thận trọng)"
        term = "Quan sát thêm"
    elif score >= 45:
        signal = "GIỮ"
        term = "Theo dõi"
    elif score >= 30:
        signal = "BÁN (thận trọng)"
        term = "Cân nhắc bán"
    else:
        signal = "BÁN"
        term = "Nên bán"

    # ========================================
    # BƯỚC 6: KIỂM TRA MÂU THUẪN CUỐI CÙNG
    # ========================================

    # Nếu tín hiệu MUA nhưng có >= 3 lý do tiêu cực
    if signal.startswith("MUA") and negative_count >= 3:
        signal = "GIỮ"
        term = "Tín hiệu mâu thuẫn - nên đợi"
        reasons.insert(0, "⚠️⚠️ CẢNH BÁO MÂU THUẪN: Nhiều tín hiệu tiêu cực")

    reason_text = "\n".join(reasons[:15])
    return signal, score, reason_text, term, details


# ============================================
# 5B: ML PREDICTION
# ============================================

def predict_trend_ml_enhanced(df, forecast_days=7):
    """Dự đoán xu hướng bằng ML"""
    if df is None or df.empty or len(df) < 100:
        return None, 0.5

    try:
        features = [
            'RSI', 'MACD', 'Stoch_K', 'ADX', 'Volume_ratio',
            'MFI', 'CCI', 'Williams_R', 'BB_width', 'ATR',
            'Aroon_up', 'Aroon_down', 'UltimateOsc', 'TRIX',
            'StochRSI', 'TSI', 'Vortex_plus', 'Vortex_minus',
            'Bull_Power', 'Bear_Power', 'AO', 'Choppiness'
        ]

        available_features = [f for f in features if f in df.columns]
        df_clean = df[available_features].dropna()

        if len(df_clean) < 50:
            return None, 0.5

        df_clean['target'] = (df.loc[df_clean.index, 'close'].shift(-forecast_days) >
                              df.loc[df_clean.index, 'close']).astype(int)
        df_clean = df_clean.dropna()

        if len(df_clean) < 50:
            return None, 0.5

        split = int(len(df_clean) * 0.8)
        X_train = df_clean[available_features].iloc[:split]
        y_train = df_clean['target'].iloc[:split]

        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)

        latest_features = df_clean[available_features].iloc[-1:].values
        prediction = model.predict(latest_features)[0]
        probability = model.predict_proba(latest_features)[0]

        trend = f"TĂNG ({forecast_days} ngày)" if prediction == 1 else f"GIẢM ({forecast_days} ngày)"
        confidence = max(probability)

        return trend, confidence
    except Exception as e:
        print(f"Error ML: {e}")
        return None, 0.5


# ============================================
# 5C: HELPER FUNCTIONS
# ============================================

def detect_candlestick_patterns(df):
    """Phát hiện mẫu nến"""
    if df is None or len(df) < 2:
        return {}

    latest = df.iloc[-1]

    patterns = {
        'Doji': latest.get('Doji', 0) == 1,
        'Hammer': latest.get('Hammer', 0) == 1,
        'Shooting Star': latest.get('Shooting_Star', 0) == 1,
        'Bullish Engulfing': latest.get('Bullish_Engulfing', 0) == 1,
        'Bearish Engulfing': latest.get('Bearish_Engulfing', 0) == 1
    }

    return {k: v for k, v in patterns.items() if v}


def simple_backtest(df, initial_capital=100000000):
    """Backtest chiến lược"""
    if df is None or df.empty or len(df) < 100:
        return None

    capital = initial_capital
    shares = 0
    trades = []

    for i in range(50, len(df)):
        current_data = df.iloc[:i + 1]
        current_data = calculate_advanced_indicators(current_data)
        signal, score, _, _, _ = generate_advanced_signal(current_data)

        current_price = df.iloc[i]['close']

        # Chỉ mua khi tín hiệu rõ ràng (>= 65 điểm)
        if signal == "MUA MẠNH" and shares == 0 and capital > current_price:
            shares = int(capital * 0.95 / current_price)
            cost = shares * current_price
            capital -= cost
            trades.append({
                'date': df.index[i],
                'action': 'BUY',
                'price': current_price,
                'shares': shares,
                'value': cost
            })

        # Bán khi có tín hiệu bán hoặc điểm < 55
        elif (signal.startswith("BÁN") or score < 55) and shares > 0:
            revenue = shares * current_price
            capital += revenue
            trades.append({
                'date': df.index[i],
                'action': 'SELL',
                'price': current_price,
                'shares': shares,
                'value': revenue
            })
            shares = 0

    if shares > 0:
        final_price = df.iloc[-1]['close']
        capital += shares * final_price
        shares = 0

    final_value = capital
    roi = ((final_value - initial_capital) / initial_capital) * 100

    return {
        'trades': trades,
        'final_value': final_value,
        'roi': roi,
        'num_trades': len(trades)
    }


print("✅ Smart Signal Logic loaded!")
# ============================================
# PHẦN 6A: MODULE HOLDING PERIOD (THÊM VÀO ĐÂY!)
# ============================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def calculate_holding_period(df, signal, score, details):
    """
    Tính toán thời gian nắm giữ tối ưu
    """
    if df is None or df.empty or len(df) < 50:
        return None

    latest = df.iloc[-1]

    # Xác định xu hướng
    trend_type = "SIDEWAY"
    trend_strength = 0

    if pd.notna(latest.get('ADX')):
        adx = latest['ADX']
        if adx > 40:
            trend_strength = 3
        elif adx > 25:
            trend_strength = 2
        elif adx > 20:
            trend_strength = 1
        else:
            trend_strength = 0

    if pd.notna(latest.get('Supertrend_direction')):
        if latest['Supertrend_direction'] == 1:
            trend_type = "TĂNG"
        elif latest['Supertrend_direction'] == -1:
            trend_type = "GIẢM"

    if pd.notna(latest.get('MA5')) and pd.notna(latest.get('MA20')) and pd.notna(latest.get('MA50')):
        if latest['MA5'] > latest['MA20'] > latest['MA50']:
            if trend_type == "SIDEWAY":
                trend_type = "TĂNG"
        elif latest['MA5'] < latest['MA20'] < latest['MA50']:
            if trend_type == "SIDEWAY":
                trend_type = "GIẢM"

    # Volatility
    volatility_level = "TRUNG BÌNH"

    if pd.notna(latest.get('ATR')) and pd.notna(latest.get('close')):
        atr_percent = (latest['ATR'] / latest['close']) * 100

        if atr_percent > 5:
            volatility_level = "RẤT CAO"
        elif atr_percent > 3:
            volatility_level = "CAO"
        elif atr_percent > 1.5:
            volatility_level = "TRUNG BÌNH"
        else:
            volatility_level = "THẤP"

    # Vị trí trong xu hướng
    trend_position = "GIỮA"

    if pd.notna(latest.get('RSI')):
        rsi = latest['RSI']
        if rsi < 30:
            trend_position = "ĐẦU"
        elif rsi > 70:
            trend_position = "CUỐI"
        elif 40 <= rsi <= 60:
            trend_position = "GIỮA"

    if pd.notna(latest.get('MACD')) and pd.notna(latest.get('MACD_signal')):
        prev = df.iloc[-2]
        if latest['MACD'] > latest['MACD_signal'] and prev['MACD'] <= prev['MACD_signal']:
            trend_position = "ĐẦU"
        elif latest['MACD'] < latest['MACD_signal'] and prev['MACD'] >= prev['MACD_signal']:
            trend_position = "CUỐI"

    # Tính thời gian
    base_days = 0
    min_days = 0
    max_days = 0
    recommended_action = ""
    reasons = []

    if trend_type == "TĂNG" and signal in ["MUA MẠNH", "MUA", "MUA (thận trọng)"]:
        if trend_position == "ĐẦU":
            if trend_strength >= 2:
                base_days = 30
                min_days = 20
                max_days = 60
                recommended_action = "NẮM GIỮ DÀI HẠN"
                reasons.append("✅ Xu hướng tăng mạnh mới bắt đầu")
            else:
                base_days = 10
                min_days = 5
                max_days = 20
                recommended_action = "NẮM GIỮ NGẮN HẠN"
                reasons.append("📈 Xu hướng tăng yếu")

        elif trend_position == "GIỮA":
            if trend_strength >= 2:
                base_days = 20
                min_days = 10
                max_days = 40
                recommended_action = "NẮM GIỮ TRUNG HẠN"
                reasons.append("✅ Xu hướng tăng ổn định")
            else:
                base_days = 7
                min_days = 3
                max_days = 14
                recommended_action = "NẮM GIỮ NGẮN"
                reasons.append("⚠️ Xu hướng không rõ ràng")

        else:
            base_days = 3
            min_days = 1
            max_days = 7
            recommended_action = "CHỐT LỜI SỚM"
            reasons.append("⚠️⚠️ Xu hướng tăng gần hết")
            reasons.append("💡 Nên chốt lời")

    elif trend_type == "GIẢM":
        base_days = 0
        min_days = 0
        max_days = 0
        recommended_action = "BÁN NGAY"
        reasons.append("🚫 Xu hướng giảm")
        reasons.append("💡 Không nên nắm giữ")

    else:
        if signal == "GIỮ":
            base_days = 5
            min_days = 3
            max_days = 10
            recommended_action = "THEO DÕI SÁT"
            reasons.append("⚠️ Thị trường sideway")
        else:
            base_days = 1
            min_days = 0
            max_days = 3
            recommended_action = "CÂN NHẮC BÁN"
            reasons.append("⚠️ Tín hiệu không rõ")

    # Điều chỉnh theo volatility
    volatility_adjustment = 1.0

    if volatility_level == "RẤT CAO":
        volatility_adjustment = 0.6
        reasons.append("⚠️ Biến động cao → Giảm thời gian")
    elif volatility_level == "CAO":
        volatility_adjustment = 0.8
    elif volatility_level == "THẤP":
        volatility_adjustment = 1.2
        reasons.append("✅ Biến động thấp → An toàn")

    base_days = int(base_days * volatility_adjustment)
    min_days = int(min_days * volatility_adjustment)
    max_days = int(max_days * volatility_adjustment)

    if score >= 80:
        reasons.append("✅✅ Điểm AI cao → Tự tin")
    elif score < 50:
        base_days = max(1, int(base_days * 0.5))
        reasons.append("⚠️ Điểm AI thấp")

    # Target & Stop Loss
    current_price = latest['close']

    if pd.notna(latest.get('ATR')):
        atr = latest['ATR']
        if trend_type == "TĂNG":
            target_price = current_price + (atr * (base_days / 5) * 1.5)
            stop_loss = current_price - (atr * 2)
        else:
            target_price = current_price
            stop_loss = current_price - (atr * 1.5)
    else:
        target_price = current_price * 1.05
        stop_loss = current_price * 0.97

    # Dates
    today = datetime.now()
    target_date_min = today + timedelta(days=min_days)
    target_date_base = today + timedelta(days=base_days)
    target_date_max = today + timedelta(days=max_days)

    # Risk level
    risk_level = "TRUNG BÌNH"
    if volatility_level in ["RẤT CAO", "CAO"] and trend_strength < 2:
        risk_level = "CAO"
    elif volatility_level == "RẤT THẤP" and trend_strength >= 2:
        risk_level = "THẤP"

    return {
        'base_days': base_days,
        'min_days': min_days,
        'max_days': max_days,
        'recommended_action': recommended_action,
        'trend_type': trend_type,
        'trend_strength': trend_strength,
        'trend_position': trend_position,
        'volatility_level': volatility_level,
        'risk_level': risk_level,
        'target_date_min': target_date_min.strftime('%d/%m/%Y'),
        'target_date_base': target_date_base.strftime('%d/%m/%Y'),
        'target_date_max': target_date_max.strftime('%d/%m/%Y'),
        'target_price': target_price,
        'stop_loss': stop_loss,
        'current_price': current_price,
        'reasons': reasons
    }


def display_holding_recommendation(holding_info):
    """Hiển thị khuyến nghị"""
    if holding_info is None:
        return

    # Background color theo action
    action_colors = {
        "NẮM GIỮ DÀI HẠN": "#00c853",
        "NẮM GIỮ TRUNG HẠN": "#66BB6A",
        "NẮM GIỮ NGẮN HẠN": "#FFA726",
        "BÁN NGAY": "#d32f2f"
    }
    color = action_colors.get(holding_info['recommended_action'], "#FFA726")

    # Container với màu nền
    st.markdown(f"""
    <style>
    .holding-container {{
        background: linear-gradient(135deg, {color} 0%, {color}DD 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        margin: 20px 0;
    }}
    </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="holding-container">', unsafe_allow_html=True)

        st.markdown("### 🕐 KHUYẾN NGHỊ THỜI GIAN NẮM GIỮ")

        # Action chính
        st.markdown(f"## {holding_info['recommended_action']}")

        # 3 cột thời gian
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Tối thiểu", f"{holding_info['min_days']} ngày")
            st.caption(holding_info['target_date_min'])

        with col2:
            st.metric("⭐ Khuyến nghị", f"{holding_info['base_days']} ngày")
            st.caption(holding_info['target_date_base'])

        with col3:
            st.metric("Tối đa", f"{holding_info['max_days']} ngày")
            st.caption(holding_info['target_date_max'])

        # Target & Stop Loss
        st.markdown("---")
        target_pct = ((holding_info['target_price'] - holding_info['current_price']) /
                      holding_info['current_price'] * 100)
        stop_pct = ((holding_info['stop_loss'] - holding_info['current_price']) /
                    holding_info['current_price'] * 100)

        col1, col2 = st.columns(2)
        col1.metric("🎯 Target", f"{holding_info['target_price']:,.0f} VND", f"{target_pct:+.1f}%")
        col2.metric("🛑 Stop Loss", f"{holding_info['stop_loss']:,.0f} VND", f"{stop_pct:.1f}%")

        # Phân tích
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric("Xu hướng", holding_info['trend_type'])
        col2.metric("Biến động", holding_info['volatility_level'])
        col3.metric("Rủi ro", holding_info['risk_level'])

        # Lý do
        st.markdown("---")
        st.markdown("**💡 Lý do:**")
        for reason in holding_info['reasons']:
            st.markdown(f"- {reason}")

        st.markdown('</div>', unsafe_allow_html=True)
# ============================================
# PHẦN 6: HÀM VẼ BIỂU ĐỒ (FIXED)
# ============================================

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import timedelta
import pandas as pd


# ============================================
# 6A: BIỂU ĐỒ GIÁ NÂNG CAO
# ============================================

def plot_advanced_chart(df, symbol, predictions=None):
    """Vẽ biểu đồ giá với candlestick và indicators"""
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Giá',
        increasing_line_color='#00c853',
        decreasing_line_color='#d32f2f'
    ))

    # Moving Averages
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MA5'],
        name='MA5',
        line=dict(color='#2196F3', width=1)
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MA20'],
        name='MA20',
        line=dict(color='#FF9800', width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MA50'],
        name='MA50',
        line=dict(color='#9C27B0', width=2)
    ))

    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=df.index, y=df['BB_upper'],
        name='BB Upper',
        line=dict(color='rgba(250,128,114,0.3)', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['BB_lower'],
        name='BB Lower',
        line=dict(color='rgba(250,128,114,0.3)', dash='dash'),
        fill='tonexty',
        fillcolor='rgba(250,128,114,0.1)'
    ))

    # Dự đoán
    if predictions is not None and len(predictions) > 0:
        future_dates = pd.date_range(
            start=df.index[-1] + timedelta(days=1),
            periods=len(predictions)
        )
        fig.add_trace(go.Scatter(
            x=future_dates, y=predictions,
            name='Dự đoán',
            line=dict(color='#FFC107', width=3, dash='dot'),
            mode='lines+markers'
        ))

    fig.update_layout(
        title=f'{symbol} - Biểu Đồ Kỹ Thuật',
        xaxis_title='Ngày',
        yaxis_title='Giá (VND)',
        template='plotly_dark',
        height=600,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


# ============================================
# 6B: BIỂU ĐỒ KHỐI LƯỢNG (FIXED - ĐÃ SỬA LỖI)
# ============================================

def plot_volume_chart(df):
    """Vẽ biểu đồ khối lượng giao dịch"""
    fig = go.Figure()

    colors = [
        '#00c853' if df['close'].iloc[i] >= df['open'].iloc[i] else '#d32f2f'
        for i in range(len(df))
    ]

    fig.add_trace(go.Bar(
        x=df.index,
        y=df['volume'],
        name='Volume',
        marker_color=colors
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Volume_MA'],
        name='Volume MA20',
        line=dict(color='#FFC107', width=2)
    ))

    fig.update_layout(
        title='Khối Lượng Giao Dịch',
        xaxis_title='Ngày',
        yaxis_title='Khối lượng',
        template='plotly_dark',
        height=300,
        hovermode='x unified'
    )  # ← ĐÃ SỬA: Thêm dấu đóng ngoặc

    return fig


# ============================================
# 6C: BIỂU ĐỒ CHỈ BÁO KỸ THUẬT
# ============================================

def plot_multi_indicators(df):
    """Vẽ nhiều chỉ báo trong subplots"""
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('RSI', 'MACD', 'Stochastic', 'ADX', 'MFI'),
        row_heights=[0.2, 0.2, 0.2, 0.2, 0.2]
    )

    # RSI
    fig.add_trace(go.Scatter(
        x=df.index, y=df['RSI'],
        name='RSI',
        line=dict(color='#2196F3')
    ), row=1, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=1, col=1)

    # MACD
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MACD'],
        name='MACD',
        line=dict(color='#2196F3')
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MACD_signal'],
        name='Signal',
        line=dict(color='#FF9800')
    ), row=2, col=1)
    fig.add_trace(go.Bar(
        x=df.index, y=df['MACD_hist'],
        name='Histogram',
        marker_color=df['MACD_hist'].apply(
            lambda x: '#00c853' if x > 0 else '#d32f2f'
        )
    ), row=2, col=1)

    # Stochastic
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Stoch_K'],
        name='%K',
        line=dict(color='#2196F3')
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Stoch_D'],
        name='%D',
        line=dict(color='#FF9800')
    ), row=3, col=1)
    fig.add_hline(y=80, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="green", row=3, col=1)

    # ADX
    fig.add_trace(go.Scatter(
        x=df.index, y=df['ADX'],
        name='ADX',
        line=dict(color='#9C27B0', width=2)
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['DI_plus'],
        name='+DI',
        line=dict(color='#00c853')
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['DI_minus'],
        name='-DI',
        line=dict(color='#d32f2f')
    ), row=4, col=1)
    fig.add_hline(y=25, line_dash="dash", line_color="white", row=4, col=1)

    # MFI
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MFI'],
        name='MFI',
        line=dict(color='#E91E63')
    ), row=5, col=1)
    fig.add_hline(y=80, line_dash="dash", line_color="red", row=5, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="green", row=5, col=1)

    fig.update_layout(
        height=1000,
        template='plotly_dark',
        showlegend=True,
        hovermode='x unified'
    )

    return fig


# ============================================
# 6D: BIỂU ĐỒ DÒNG TIỀN
# ============================================

def plot_money_flow_chart(df):
    """Vẽ biểu đồ phân tích dòng tiền"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Money Flow Index (MFI)', 'On-Balance Volume (OBV)'),
        row_heights=[0.5, 0.5]
    )

    # MFI
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MFI'],
        name='MFI',
        line=dict(color='#E91E63', width=2),
        fill='tozeroy',
        fillcolor='rgba(233, 30, 99, 0.2)'
    ), row=1, col=1)
    fig.add_hline(y=80, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="green", row=1, col=1)

    # OBV
    obv_color = '#00c853' if df['OBV'].iloc[-1] > df['OBV'].iloc[-20] else '#d32f2f'
    fig.add_trace(go.Scatter(
        x=df.index, y=df['OBV'],
        name='OBV',
        line=dict(color=obv_color, width=2)
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['OBV'].rolling(window=20).mean(),
        name='OBV MA20',
        line=dict(color='#FFC107', width=2, dash='dash')
    ), row=2, col=1)

    fig.update_layout(
        title='Phân Tích Dòng Tiền',
        template='plotly_dark',
        height=600,
        hovermode='x unified'
    )

    return fig


# ============================================
# 6E: BIỂU ĐỒ PHÂN BỔ NGÀNH
# ============================================

def plot_sector_distribution(sector_data):
    """Vẽ biểu đồ phân bổ theo ngành"""
    sectors = list(sector_data.keys())
    percentages = [sector_data[s]['percentage'] for s in sectors]
    counts = [sector_data[s]['count'] for s in sectors]

    fig = go.Figure(data=[go.Pie(
        labels=sectors,
        values=percentages,
        hole=0.4,
        textinfo='label+percent',
        textposition='outside',
        marker=dict(
            colors=px.colors.qualitative.Set3,
            line=dict(color='#000000', width=2)
        ),
        hovertemplate='<b>%{label}</b><br>' +
                      'Tỷ trọng: %{value:.1f}%<br>' +
                      'Số mã: %{text}<br>' +
                      '<extra></extra>',
        text=counts
    )])

    fig.update_layout(
        title='Phân Bổ Vốn Hóa Theo Ngành',
        template='plotly_dark',
        height=500,
        showlegend=True
    )

    return fig


# ============================================
# 6F: BIỂU ĐỒ SO SÁNH CỔ PHIẾU
# ============================================

def plot_comparison_chart(stocks_data, period='6mo'):
    """Vẽ biểu đồ so sánh nhiều cổ phiếu (normalized)"""
    fig = go.Figure()

    for symbol, df in stocks_data.items():
        if df is not None and not df.empty:
            # Chuẩn hóa giá về 100
            normalized = (df['close'] / df['close'].iloc[0]) * 100
            fig.add_trace(go.Scatter(
                x=df.index,
                y=normalized,
                name=symbol,
                mode='lines',
                line=dict(width=2)
            ))

    fig.update_layout(
        title='So Sánh Biến Động Giá (Chuẩn hóa = 100)',
        xaxis_title='Ngày',
        yaxis_title='Giá trị (%)',
        template='plotly_dark',
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


# ============================================
# 6G: BIỂU ĐỒ CORRELATION MATRIX
# ============================================

def plot_correlation_matrix(stocks_data):
    """Vẽ ma trận tương quan giữa các cổ phiếu"""
    close_prices = pd.DataFrame({
        symbol: df['close']
        for symbol, df in stocks_data.items()
        if df is not None and not df.empty
    })

    corr = close_prices.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdYlGn',
        zmid=0,
        text=corr.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))

    fig.update_layout(
        title='Ma Trận Tương Quan',
        template='plotly_dark',
        height=500,
        xaxis_title='',
        yaxis_title=''
    )

    return fig


print("✅ Chart plotting functions loaded (FIXED)")

# ============================================
# PHẦN 7: UI CHÍNH - GIAO DIỆN HOÀN CHỈNH (FIXED)
# ============================================

import streamlit as st
import pandas as pd
from datetime import datetime

# ============================================
# 7A: HEADER & TITLE
# ============================================

# Lấy thống kê visitor
visitor_stats = get_visitor_stats()

st.title("📊 PHÂN TÍCH CỔ PHIẾU VIỆT NAM PRO")
st.markdown(f"""
### {len(ALL_VN_STOCKS)}+ mã cổ phiếu | 20+ chỉ báo | AI/ML Prediction | Phân tích dòng tiền
""")

# ============================================
# 7B: SIDEBAR - ĐIỀU KHIỂN
# ============================================

with st.sidebar:
    st.header("⚙️ ĐIỀU KHIỂN")

    # Chế độ phân tích
    mode = st.radio("Chọn chế độ", [
        "🔍 Phân tích chi tiết",
        "🚀 Quét nhanh",
        "📊 So sánh",
        "🤖 AI Prediction",
        "📈 Backtesting",
        "💰 Phân tích dòng tiền"
    ])

    st.markdown("---")

    # ============================================
    # CẤU HÌNH THEO CHẾ ĐỘ
    # ============================================

    if mode == "🔍 Phân tích chi tiết":
        st.subheader("Tìm kiếm")
        search_term = st.text_input("🔎 Tìm mã nhanh", "")

        if search_term:
            filtered_stocks = [s for s in ALL_VN_STOCKS if search_term.upper() in s]
        else:
            sector = st.selectbox("Chọn ngành", ['Tất cả'] + list(VN_STOCKS_BY_SECTOR.keys()))
            filtered_stocks = ALL_VN_STOCKS if sector == 'Tất cả' else VN_STOCKS_BY_SECTOR.get(sector, [])

        symbol = st.selectbox("Chọn mã", sorted(filtered_stocks))
        period = st.selectbox("Thời gian", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)

        st.markdown("---")
        show_prediction = st.checkbox("🔮 Dự đoán giá", value=True)
        if show_prediction:
            pred_days = st.slider("Số ngày dự đoán", 3, 30, 7)
        else:
            pred_days = 7

        show_ml_trend = st.checkbox("🤖 ML Trend", value=True)
        show_money_flow = st.checkbox("💰 Phân tích dòng tiền", value=True)

    elif mode == "🚀 Quét nhanh":
        scan_mode = st.radio("Quét", ["Theo ngành", "Top 100", "Toàn bộ"])

        if scan_mode == "Theo ngành":
            sector = st.selectbox("Chọn ngành", list(VN_STOCKS_BY_SECTOR.keys()))
            stocks_to_scan = VN_STOCKS_BY_SECTOR[sector]
        elif scan_mode == "Top 100":
            stocks_to_scan = ALL_VN_STOCKS[:100]
        else:
            stocks_to_scan = ALL_VN_STOCKS

        min_score = st.slider("Điểm tối thiểu", 50, 95, 70)
        max_results = st.slider("Số kết quả", 10, 100, 20)

    elif mode == "📊 So sánh":
        compare_symbols = st.multiselect("Chọn mã (tối đa 5)", ALL_VN_STOCKS, max_selections=5)
        period = st.selectbox("Thời gian", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)

    elif mode == "🤖 AI Prediction":
        sector = st.selectbox("Chọn ngành", list(VN_STOCKS_BY_SECTOR.keys()))
        symbol = st.selectbox("Chọn mã", VN_STOCKS_BY_SECTOR[sector])
        pred_days = st.slider("Số ngày dự đoán", 7, 30, 14)

    elif mode == "📈 Backtesting":
        sector = st.selectbox("Chọn ngành", list(VN_STOCKS_BY_SECTOR.keys()))
        symbol = st.selectbox("Chọn mã", VN_STOCKS_BY_SECTOR[sector])
        initial_capital = st.number_input("Vốn ban đầu (VND)",
                                          min_value=10000000,
                                          value=100000000,
                                          step=10000000)

    else:  # Phân tích dòng tiền
        st.subheader("Phân tích dòng tiền")
        analysis_sectors = st.multiselect("Chọn ngành",
                                          list(VN_STOCKS_BY_SECTOR.keys()),
                                          default=[list(VN_STOCKS_BY_SECTOR.keys())[0]])

    st.markdown("---")
    st.info("""
💡 **Điểm số AI:**
- ≥80: MUA MẠNH
- 70-79: MUA
- 60-69: MUA thận trọng
- 45-59: GIỮ
- 35-44: BÁN thận trọng
- <35: BÁN
    """)

    st.markdown("---")
    popular_stocks = get_popular_stocks(5)
    if popular_stocks:
        st.markdown("**🔥 Top tìm kiếm:**")
        for stock, count in popular_stocks[:5]:
            st.text(f"{stock}: {count} lượt")

# ============================================
# 7C: NỘI DUNG CHÍNH
# ============================================

# MODE 1: PHÂN TÍCH CHI TIẾT
if mode == "🔍 Phân tích chi tiết":
    st.header(f"📊 PHÂN TÍCH CHI TIẾT: {symbol}")

    track_stock_search(symbol)

    with st.spinner(f"Đang tải dữ liệu {symbol}..."):
        df, info = get_stock_data(symbol, period=period)

    if df is not None and not df.empty:
        df = calculate_advanced_indicators(df)
        signal, score, reason, term, details = generate_advanced_signal(df, info)
        latest = df.iloc[-1]

        predictions = None
        pred_confidence = 0
        if show_prediction:
            predictions, pred_confidence = predict_future_price_enhanced(df, pred_days)

        ml_trend = "N/A"
        ml_confidence = 0
        if show_ml_trend:
            ml_trend, ml_confidence = predict_trend_ml_enhanced(df, pred_days)

        money_flow = None
        if show_money_flow:
            money_flow = calculate_money_flow(df)

        # Metrics
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        price_change = ((latest['close'] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100) if len(df) > 1 else 0

        col1.metric("💰 Giá", f"{latest['close']:,.0f}", f"{price_change:+.2f}%")
        col2.metric("📊 Volume", f"{latest['volume'] / 1000:.0f}K")
        col3.metric("⭐ Điểm AI", f"{score}/100", f"{score - 50:+.0f}")
        col4.metric("🎯 Khuyến nghị", term)
        col5.metric("📈 RSI", f"{latest['RSI']:.1f}")
        col6.metric("💪 ADX", f"{latest['ADX']:.1f}")

        # Signal
        if signal.startswith("MUA"):
            st.markdown(f'<div class="buy-signal">🟢 {signal} - Điểm: {score}/100</div>', unsafe_allow_html=True)
        elif signal.startswith("BÁN"):
            st.markdown(f'<div class="sell-signal">🔴 {signal} - Điểm: {score}/100</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="hold-signal">🟡 {signal} - Điểm: {score}/100</div>', unsafe_allow_html=True)

        # ML Prediction
        if ml_trend != "N/A":
            trend_color = "#00c853" if ml_trend == "TĂNG" else "#d32f2f"
            st.markdown(f"""
            <div class="prediction-box">
                <h3>🤖 Dự đoán AI/ML</h3>
                <p style="font-size:24px;">
                    Xu hướng: <span style="color:{trend_color}; font-weight:bold;">{ml_trend}</span>
                    | Độ tin cậy: <b>{ml_confidence * 100:.1f}%</b>
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Price prediction
        if predictions is not None and len(predictions) > 0:
            pred_change = ((predictions[-1] - latest['close']) / latest['close']) * 100
            pred_color = "#00c853" if pred_change > 0 else "#d32f2f"
            st.markdown(f"""
            <div class="prediction-box">
                <h3>🔮 Dự đoán giá {pred_days} ngày</h3>
                <p style="font-size:20px;">
                    Giá dự kiến: <b>{predictions[-1]:,.0f} VND</b> | 
                    Thay đổi: <span style="color:{pred_color};">{pred_change:+.2f}%</span><br>
                    Độ tin cậy: <b>{pred_confidence * 100:.1f}%</b>
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Money Flow
        if money_flow:
            st.markdown(f"""
            <div class="money-flow-box">
                <h3>💰 Phân tích dòng tiền</h3>
                <p style="font-size:18px;">
                    MFI: <b>{money_flow['MFI']:.1f}</b> | 
                    Trạng thái: <span style="color:{money_flow['color']};">{money_flow['status']}</span><br>
                    OBV: <b>{money_flow['obv_trend']}</b> | 
                    Volume: <b>{money_flow['volume_change']:+.1f}%</b>
                </p>
            </div>
            """, unsafe_allow_html=True)

        # HOLDING PERIOD RECOMMENDATION
        holding_info = calculate_holding_period(df, signal, score, details)
        if holding_info:
            display_holding_recommendation(holding_info)

        st.markdown("---")
        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Biểu đồ giá", "📊 Chỉ báo", "📝 Phân tích", "💼 Fundamental", "🎯 Mức giá"
        ])

        with tab1:
            st.plotly_chart(plot_advanced_chart(df, symbol, predictions), use_container_width=True)
            st.plotly_chart(plot_volume_chart(df), use_container_width=True)

        with tab2:
            st.plotly_chart(plot_multi_indicators(df), use_container_width=True)
            if show_money_flow:
                st.plotly_chart(plot_money_flow_chart(df), use_container_width=True)

        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**🎯 Tín hiệu: {signal}**")
                st.markdown(f"**⭐ Điểm: {score}/100**")
                st.markdown(f"**⏰ Khuyến nghị: {term}**")

                # Hiển thị VETO nếu có
                if '🚫' in reason or 'VETO' in reason:
                    st.error("⚠️⚠️ CÓ TÍN HIỆU CẢNH BÁO NGHIÊM TRỌNG!")

                st.markdown("---")
                st.markdown("**💡 Chi tiết điểm:**")
                for key, value in details.items():
                    if isinstance(value, (int, float)):
                        st.text(f"{key}: {value} điểm")
                    else:
                        st.text(f"{key}: {value}")
            with col2:
                st.markdown("**🔍 Lý do chi tiết:**")

                # Phân loại lý do
                positive_reasons = [r for r in reason.split('\n') if '✅' in r or '💰' in r]
                negative_reasons = [r for r in reason.split('\n') if '⚠️' in r or '🚫' in r]
                neutral_reasons = [r for r in reason.split('\n') if '📈' in r or '📊' in r]

                if negative_reasons:
                    st.markdown("**🔴 Tín hiệu tiêu cực:**")
                    for r in negative_reasons:
                        st.text(r)

                if positive_reasons:
                    st.markdown("**🟢 Tín hiệu tích cực:**")
                    for r in positive_reasons:
                        st.text(r)

                if neutral_reasons:
                    st.markdown("**🟡 Tín hiệu trung lập:**")
                    for r in neutral_reasons:
                        st.text(r)

        with tab4:
            st.subheader("💼 Dữ liệu Fundamental")
            if info and isinstance(info, dict) and len(info) > 5:
                col1, col2, col3 = st.columns(3)

                with col1:
                    market_cap = info.get('marketCap')
                    if market_cap and market_cap > 0:
                        st.metric("Market Cap", f"{market_cap / 1e9:.2f}B VND")
                    else:
                        st.metric("Market Cap", "N/A")

                    pe = info.get('trailingPE') or info.get('forwardPE')
                    st.metric("P/E Ratio", f"{pe:.2f}" if pe and pe > 0 else "N/A")

                    peg = info.get('pegRatio')
                    st.metric("PEG Ratio", f"{peg:.2f}" if peg and peg > 0 else "N/A")

                with col2:
                    eps = info.get('trailingEps') or info.get('forwardEps')
                    st.metric("EPS", f"{eps:.2f}" if eps else "N/A")

                    roe = info.get('returnOnEquity')
                    st.metric("ROE", f"{roe * 100:.2f}%" if roe else "N/A")

                    roa = info.get('returnOnAssets')
                    st.metric("ROA", f"{roa * 100:.2f}%" if roa else "N/A")

                with col3:
                    profit_margin = info.get('profitMargins')
                    st.metric("Profit Margin", f"{profit_margin * 100:.2f}%" if profit_margin else "N/A")

                    dividend = info.get('dividendYield')
                    st.metric("Dividend Yield", f"{dividend * 100:.2f}%" if dividend else "N/A")

                    debt_equity = info.get('debtToEquity')
                    st.metric("Debt/Equity", f"{debt_equity:.2f}" if debt_equity else "N/A")

                st.info("ℹ️ Lưu ý: Dữ liệu fundamental của cổ phiếu Việt Nam có thể không đầy đủ trên Yahoo Finance")
            else:
                st.warning("⚠️ Không có đủ dữ liệu fundamental từ Yahoo Finance cho mã này")
                st.info("💡 Bạn có thể tham khảo thêm tại: vndirect.com.vn hoặc cafef.vn")

        with tab5:
            sr = get_support_resistance(df)
            if sr:
                col1, col2, col3 = st.columns(3)
                col1.metric("🔴 Kháng cự 1", f"{sr['resistance'][0]:,.0f}")
                col2.metric("💰 Hiện tại", f"{sr['current']:,.0f}")
                col3.metric("🟢 Hỗ trợ 1", f"{sr['support'][0]:,.0f}")

                st.markdown("---")
                st.markdown("**📊 Các mức giá quan trọng:**")
                fib_df = pd.DataFrame({
                    'Loại': ['Kháng cự 3', 'Kháng cự 2', 'Kháng cự 1', 'Giá hiện tại', 'Hỗ trợ 1', 'Hỗ trợ 2',
                             'Hỗ trợ 3'],
                    'Giá': [
                        f"{sr['resistance'][-1]:,.0f}" if len(sr['resistance']) > 2 else "N/A",
                        f"{sr['resistance'][1]:,.0f}" if len(sr['resistance']) > 1 else "N/A",
                        f"{sr['resistance'][0]:,.0f}",
                        f"{sr['current']:,.0f}",
                        f"{sr['support'][0]:,.0f}",
                        f"{sr['support'][1]:,.0f}" if len(sr['support']) > 1 else "N/A",
                        f"{sr['support'][-1]:,.0f}" if len(sr['support']) > 2 else "N/A"
                    ]
                })
                st.dataframe(fib_df, hide_index=True, use_container_width=True)
    else:
        st.error(f"❌ Không thể tải dữ liệu {symbol}. Vui lòng thử mã khác.")

# MODE 2: QUÉT NHANH
elif mode == "🚀 Quét nhanh":
    st.header("🚀 QUÉT NHANH CỔ PHIẾU")

    st.info(f"📊 Sẽ quét {len(stocks_to_scan)} mã cổ phiếu | Điểm tối thiểu: {min_score}")

    if st.button("🔍 BẮT ĐẦU QUÉT", type="primary", use_container_width=True):
        results = []
        progress = st.progress(0)
        status = st.empty()

        for idx, sym in enumerate(stocks_to_scan):
            status.text(f"Đang quét {sym}... ({idx + 1}/{len(stocks_to_scan)})")

            try:
                df, _ = get_stock_data(sym, period='6mo')
                if df is not None and not df.empty:
                    df = calculate_advanced_indicators(df)
                    signal, score, _, term, _ = generate_advanced_signal(df)

                    if score >= min_score:
                        latest = df.iloc[-1]
                        ml_trend, ml_conf = predict_trend_ml_enhanced(df, 7)

                        results.append({
                            'Mã': sym,
                            'Tín hiệu': signal,
                            'Điểm': score,
                            'Giá': latest['close'],
                            'RSI': latest['RSI'],
                            'ADX': latest['ADX'],
                            'ML': ml_trend,
                            'Tin cậy': f"{ml_conf * 100:.0f}%"
                        })

                        if len(results) >= max_results:
                            break
            except:
                pass

            progress.progress((idx + 1) / len(stocks_to_scan))

        progress.empty()
        status.empty()

        if results:
            result_df = pd.DataFrame(results).sort_values('Điểm', ascending=False)
            st.success(f"✅ Tìm thấy {len(result_df)} cổ phiếu tiềm năng!")
            st.dataframe(result_df, use_container_width=True, height=600)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🎯 Điểm TB", f"{result_df['Điểm'].mean():.1f}")
            col2.metric("⭐ Cao nhất", f"{result_df['Điểm'].max():.0f}")
            col3.metric("📊 MUA", len(result_df[result_df['Tín hiệu'].str.contains('MUA')]))
            col4.metric("🤖 ML TĂNG", len(result_df[result_df['ML'] == 'TĂNG']))

            csv = result_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button("📥 Tải CSV", csv, f"scan_{datetime.now():%Y%m%d_%H%M}.csv", "text/csv")
        else:
            st.warning("⚠️ Không tìm thấy cổ phiếu nào đạt tiêu chí")

# MODE 3: SO SÁNH
elif mode == "📊 So sánh":
    st.header("📊 SO SÁNH CỔ PHIẾU")

    if compare_symbols and len(compare_symbols) >= 2:
        with st.spinner("Đang tải dữ liệu..."):
            stocks_data = {}
            comparison_data = []

            for sym in compare_symbols:
                df, _ = get_stock_data(sym, period=period)
                if df is not None and not df.empty:
                    df = calculate_advanced_indicators(df)
                    signal, score, _, term, _ = generate_advanced_signal(df)
                    latest = df.iloc[-1]
                    ml_trend, ml_conf = predict_trend_ml_enhanced(df, 7)

                    stocks_data[sym] = df
                    comparison_data.append({
                        'Mã': sym,
                        'Giá': latest['close'],
                        'RSI': latest['RSI'],
                        'MACD': latest['MACD'],
                        'ADX': latest['ADX'],
                        'Điểm': score,
                        'Tín hiệu': signal,
                        'ML': ml_trend
                    })

        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            st.dataframe(comp_df, use_container_width=True)

            st.subheader("📈 So sánh biến động giá")
            st.plotly_chart(plot_comparison_chart(stocks_data, period), use_container_width=True)

            if len(stocks_data) >= 2:
                st.subheader("🔗 Ma trận tương quan")
                st.plotly_chart(plot_correlation_matrix(stocks_data), use_container_width=True)
    else:
        st.info("ℹ️ Vui lòng chọn ít nhất 2 mã để so sánh")

# MODE 4: AI PREDICTION
elif mode == "🤖 AI Prediction":
    st.header(f"🤖 AI PREDICTION: {symbol}")

    with st.spinner("Đang phân tích..."):
        df, _ = get_stock_data(symbol, period='1y')

    if df is not None and not df.empty:
        df = calculate_advanced_indicators(df)
        signal, score, _, _, _ = generate_advanced_signal(df)
        latest = df.iloc[-1]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("💰 Giá", f"{latest['close']:,.0f}")
        col2.metric("⭐ Điểm AI", f"{score}/100")
        col3.metric("📈 RSI", f"{latest['RSI']:.1f}")
        col4.metric("💪 ADX", f"{latest['ADX']:.1f}")

        ml_trend, ml_conf = predict_trend_ml_enhanced(df, pred_days)
        trend_color = "#00c853" if "TĂNG" in ml_trend else "#d32f2f"

        st.markdown(f"""
        <div class="prediction-box">
            <h2>🤖 Machine Learning Analysis</h2>
            <p style="font-size:28px;">
                Xu hướng: <span style="color:{trend_color};">{ml_trend}</span>
                | Độ tin cậy: <b>{ml_conf * 100:.1f}%</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

        predictions, pred_confidence = predict_future_price_enhanced(df, pred_days)
        if predictions is not None:
            pred_change = ((predictions[-1] - latest['close']) / latest['close']) * 100
            pred_color = "#00c853" if pred_change > 0 else "#d32f2f"

            st.markdown(f"""
            <div class="prediction-box">
                <h2>🔮 Dự đoán giá {pred_days} ngày</h2>
                <p style="font-size:24px;">
                    Giá dự kiến: <b>{predictions[-1]:,.0f} VND</b> | 
                    <span style="color:{pred_color};">{pred_change:+.2f}%</span><br>
                    Độ tin cậy: <b>{pred_confidence * 100:.1f}%</b>
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.plotly_chart(plot_advanced_chart(df.tail(90), symbol, predictions), use_container_width=True)
    else:
        st.error("❌ Không thể tải dữ liệu")

# MODE 5: BACKTESTING
elif mode == "📈 Backtesting":
    st.header(f"📈 BACKTESTING: {symbol}")

    with st.spinner("Đang chạy backtest..."):
        df, _ = get_stock_data(symbol, period='2y')

    if df is not None and not df.empty:
        df = calculate_advanced_indicators(df)
        backtest_results = simple_backtest(df, initial_capital)

        if backtest_results:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("💰 Vốn đầu", f"{initial_capital:,.0f}")
            col2.metric("💵 Giá trị cuối", f"{backtest_results['final_value']:,.0f}")
            col3.metric("📈 ROI", f"{backtest_results['roi']:.2f}%")
            col4.metric("🔄 Giao dịch", backtest_results['num_trades'])

            buy_hold = ((df.iloc[-1]['close'] - df.iloc[50]['close']) / df.iloc[50]['close']) * 100

            col1, col2 = st.columns(2)
            col1.metric("🤖 Chiến lược AI", f"{backtest_results['roi']:.2f}%")
            col2.metric("🎯 Buy & Hold", f"{buy_hold:.2f}%")

            if backtest_results['roi'] > buy_hold:
                st.success(f"✅ AI tốt hơn: +{(backtest_results['roi'] - buy_hold):.2f}%")
            else:
                st.warning(f"⚠️ Buy & Hold tốt hơn: +{(buy_hold - backtest_results['roi']):.2f}%")

            if backtest_results['trades']:
                st.subheader("📋 Lịch sử giao dịch")
                trades_df = pd.DataFrame(backtest_results['trades'])
                trades_df['date'] = pd.to_datetime(trades_df['date']).dt.strftime('%Y-%m-%d')
                st.dataframe(trades_df, use_container_width=True, height=400)
    else:
        st.error("❌ Không thể tải dữ liệu")

# MODE 6: PHÂN TÍCH DÒNG TIỀN
else:
    st.header("💰 PHÂN TÍCH DÒNG TIỀN THỊ TRƯỜNG")

    if st.button("🔍 BẮT ĐẦU PHÂN TÍCH", type="primary"):
        all_stocks = []
        for sector in analysis_sectors:
            all_stocks.extend(VN_STOCKS_BY_SECTOR[sector])

        progress = st.progress(0)
        status = st.empty()
        results = []

        for idx, sym in enumerate(all_stocks[:50]):  # Giới hạn 50 mã
            status.text(f"Đang phân tích {sym}...")

            try:
                df, _ = get_stock_data(sym, period='3mo')
                if df is not None and not df.empty:
                    df = calculate_advanced_indicators(df)
                    money_flow = calculate_money_flow(df)
                    latest = df.iloc[-1]

                    results.append({
                        'Mã': sym,
                        'Giá': latest['close'],
                        'MFI': money_flow['MFI'],
                        'Trạng thái': money_flow['status'],
                        'OBV': money_flow['obv_trend'],
                        'Volume': f"{money_flow['volume_change']:+.1f}%"
                    })
            except:
                pass

            progress.progress((idx + 1) / min(len(all_stocks), 50))

        progress.empty()
        status.empty()

        if results:
            result_df = pd.DataFrame(results)
            st.success(f"✅ Phân tích {len(result_df)} mã cổ phiếu")
            st.dataframe(result_df, use_container_width=True, height=600)

            st.subheader("📊 Phân phối MFI")
            mfi_ranges = {
                'Quá bán (<30)': len(result_df[result_df['MFI'] < 30]),
                'Trung lập (30-70)': len(result_df[(result_df['MFI'] >= 30) & (result_df['MFI'] <= 70)]),
                'Quá mua (>70)': len(result_df[result_df['MFI'] > 70])
            }

            col1, col2, col3 = st.columns(3)
            col1.metric("💰 Quá bán", mfi_ranges['Quá bán (<30)'])
            col2.metric("📊 Trung lập", mfi_ranges['Trung lập (30-70)'])
            col3.metric("⚠️ Quá mua", mfi_ranges['Quá mua (>70)'])
        else:
            st.warning("⚠️ Không có dữ liệu")

# ============================================
# 7D: FOOTER
# ============================================

st.markdown("---")
st.markdown(f"""
<div class="footer-stats">
    <div style="font-size:14px; font-weight:bold; margin-bottom:8px;">📊 THỐNG KÊ</div>
    <div>👥 Online: <b>{visitor_stats['online_now']}</b></div>
    <div>📈 Tổng truy cập: <b>{visitor_stats['total_visits']}</b></div>
    <div>🔍 Tổng tìm kiếm: <b>{visitor_stats['total_searches']}</b></div>
    <div>💹 Tổng mã: <b>{len(ALL_VN_STOCKS)}</b></div>
    <div style="margin-top:8px; font-size:10px; opacity:0.7;">
        ⏰ {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}
    </div>
</div>
""", unsafe_allow_html=True)

st.warning("""
⚠️ **LƯU Ý QUAN TRỌNG:**
- Đây là công cụ hỗ trợ phân tích, KHÔNG phải lời khuyên đầu tư
- Kết quả dự đoán chỉ mang tính tham khảo
- Dữ liệu fundamental có thể không đầy đủ cho cổ phiếu VN
- Luôn tự nghiên cứu và chịu trách nhiệm với quyết định của mình
""")

st.info("""
💡 **NGUỒN DỮ LIỆU:**
- Giá & Volume: Yahoo Finance
- Fundamental: Yahoo Finance (có thể thiếu cho cổ phiếu VN)
- Khuyến nghị: Tham khảo thêm tại vndirect.com.vn, cafef.vn, tcbs.com.vn
""")

print("✅ App UI loaded successfully!")
