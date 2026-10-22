# folder = "/project/rrg-mdiamond/data/CDMS/CUTE/R37/"
folder = "/fs/ddn/sdf/group/supercdms/data/CDMS/CUTE/R37/"
folder = "/sdf/data/supercdms/data/CDMS/CUTE/R37/"
folder_raw = f"{folder}/Raw/"

process_version = "CUTE_T3_Calibration_v3.0.0_V05-03_P0.3.3"
process_version = "CUTE_T3_Calibration_NxMamps_v3.1.0_V05-04_P0.3.4"
folder_proc = f"{folder}/Processed/Tests/{process_version}/Submerged/{process_version}"


EK=	10.367
EL=	1.299
EM=	0.1595
V = 50
w = 3 # eV /pair
NTL = V/w+1


chs = ['PFS1', 'PCS1', 'PDS1', 'PBS1', 'PES1', 'PAS1', 'PFS2', 'PCS2', 'PBS2', 'PES2', 'PDS2', 'PAS2']
chs_rings = {1: ["F"], 2: ["C","D","E"], 3: ["B"], 4: ["A"]}
chs_ring1 = ["F"]
chs_ring2 = ["C","D","E"]
chs_ring3 = ["B"]
chs_ring4 = ["A"]
chs_ring3all = ["A","B"]
chs_sides = [1,2]
ch_up = 1
ch_down = 2

def get_rq_rings(rq_dict, rq_name, weights = None):
    if weights is None:
        ring1 = [rq_dict[f"{rq_name}_P{ich}S{iside}"] for ich in chs_ring1 for iside in [1,2]]
        ring2 = [rq_dict[f"{rq_name}_P{ich}S{iside}"] for ich in chs_ring2 for iside in [1,2]]
        ring3 = [rq_dict[f"{rq_name}_P{ich}S{iside}"] for ich in chs_ring3all for iside in [1,2]]
    else:
        ring1 = [rq_dict[f"{rq_name}_P{ich}S{iside}"]*weights[f"P{ich}S{iside}"] for ich in chs_ring1 for iside in [1,2]]
        ring2 = [rq_dict[f"{rq_name}_P{ich}S{iside}"]*weights[f"P{ich}S{iside}"] for ich in chs_ring2 for iside in [1,2]]
        ring3 = [rq_dict[f"{rq_name}_P{ich}S{iside}"]*weights[f"P{ich}S{iside}"] for ich in chs_ring3all for iside in [1,2]]        
    return ring1, ring2, ring3

def get_rq_rings_all(rq_dict, rq_name, weights = None):
    if weights is None:
        ring1 = [rq_dict[f"{rq_name}_P{ich}S{iside}"] for ich in chs_ring1 for iside in [1,2]]
        ring2 = [rq_dict[f"{rq_name}_P{ich}S{iside}"] for ich in chs_ring2 for iside in [1,2]]
        ring3 = [rq_dict[f"{rq_name}_P{ich}S{iside}"] for ich in chs_ring3 for iside in [1,2]]
        ring4 = [rq_dict[f"{rq_name}_P{ich}S{iside}"] for ich in chs_ring4 for iside in [1,2]]
    else:
        ring1 = [rq_dict[f"{rq_name}_P{ich}S{iside}"]*weights[f"P{ich}S{iside}"] for ich in chs_ring1 for iside in [1,2]]
        ring2 = [rq_dict[f"{rq_name}_P{ich}S{iside}"]*weights[f"P{ich}S{iside}"] for ich in chs_ring2 for iside in [1,2]]
        ring3 = [rq_dict[f"{rq_name}_P{ich}S{iside}"]*weights[f"P{ich}S{iside}"] for ich in chs_ring3 for iside in [1,2]]
        ring4 = [rq_dict[f"{rq_name}_P{ich}S{iside}"]*weights[f"P{ich}S{iside}"] for ich in chs_ring4 for iside in [1,2]]        
    return ring1, ring2, ring3, ring4


# Relative calibration factors for each channels from CUTE Ge Calibration Note
rcal_chs = ["PAS1","PBS1","PCS1","PDS1","PES1","PFS1","PAS2","PBS2","PCS2","PDS2","PES2","PFS2"]
# rcal_vals = ["1.2094","1.0583","0.9029","1.1162","0.8772","0.8750","1.2079","1.0427","0.8453","1.1143","0.8824","0.8684"]
# CUTE V4.0 Z1 50V , https://confluence.slac.stanford.edu/spaces/CDMS/pages/532886292/Processing+V4.0.0#Voltage-52507
rcal_vals = [1.2861 , 1.1154 , 0.9602,  1.0184 , 0.8729,  0.7997 , 1.2545,  1.0889 , 0.7989,  1.0332,  0.8984 , 0.8735]
weights_dict = {key:float(val) for key, val in zip(rcal_chs, rcal_vals)}
weights  = [weights_dict[key] for key in chs]

# # CUTE final??
# weights_dict= { "PAS1":1.2861, "PBS1":1.1154, "PCS1":0.9602, "PDS1":1.0184,	"PES1":0.8729, "PFS1":0.7997,	
#                "PAS2":1.2545, "PBS2":1.0889, "PCS2":0.7989, "PDS2":1.0332,	"PES2":0.8984, "PFS2":0.8735}
# weights  = [weights_dict[key] for key in chs]


# Toms weights, ordered by "chs"
# weights_ordered = [0.74730928,0.48400328,1.09677341,0.58642934,0.67874954,1.55048904,0.9903883,1.00895485,0.63867272,1.04109789,1.42231509,2.27521834]    
# weights_cal = [1.55048904, 0.58642934, 0.48400328, 1.09677341, 0.67874954, 0.74730928, 2.27521834, 0.63867272, 1.00895485, 1.42231509, 1.04109789, 0.9903883 ]
# joblib.dump({"weights_ordered":weights_ordered}, "data/final_psd_and_templates/weights_ordered_tom.joblib")

series_all = [
'23231216_013604', '23231216_043946', '23231216_100125', '23231216_145300', '23231216_182937',
'23231216_194929', '23231216_211119', '23231216_233807', '23231217_171613', # 23231217_135018
'23231217_212512', '23231218_093255', '23231218_152721', '23231218_190035', '23231218_223530',
'23231219_034952', '23231219_110331', '23231219_184002', '23231216_115012', '23231220_012745']
