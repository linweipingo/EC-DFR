import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)

phychem_file="data0825/data/drugfeature2_phychem_extract/drugfeature2_phychem_extract.csv"
# phychem_file="data0825/data/drugfeature2_phychem_extract\drugfeature2_phychem_normalize.csv"
finger_file="data0825/data/drugfeature1_finger_extract.csv"
cell_line_path="data0825/data/drugfeature3_express_extract/"
cell_line_files=["A-673","A375","A549","HCT116","HS 578T","HT29","LNCAP","LOVO","MCF7","PC-3","RKO","SK-MEL-28","SW-620","VCAP"]
extract_file="data0825/data/drugdrug_extract.csv"
cell_line_feature="data0825/data/cell-line-feature_express_extract.csv"


def load_data(cell_line_name="all",score="S"):
    '''
    cell_line_name: control which cell lines to load
    acceptable parameter types:list or str
        if "all"(default), load all cell lines
        if "cell line name"(e.g., "HT29"), load this cell line
        if list (e.g., ["HT29","A375"]), load "HT29" and "A375"
    '''
    extract=pd.read_csv(extract_file,usecols=[3,4,5,6,7,8,9,10,11])
    phychem=pd.read_csv(phychem_file)
    finger=pd.read_csv(finger_file)
    cell_feature=pd.read_csv(cell_line_feature)
    column_name=list(finger.columns)
    column_name[0]="drug_id"
    finger.columns=column_name

    if cell_line_name=="all":
        all_express={cell_line:pd.read_csv("{}{}.csv".format(cell_line_path,cell_line)) for cell_line in cell_line_files}
    elif type(cell_line_name) is list:
        all_express={cell_line:pd.read_csv("{}{}.csv".format(cell_line_path,cell_line)) for cell_line in cell_line_name}
    elif type(cell_line_name) is str:
        all_express={cell_line_name:pd.read_csv("{}{}.csv".format(cell_line_path,cell_line_name))}
    else:
        raise ValueError("Invalid parameter: {}".format(cell_line_name))
    
    drug_comb=None
    if cell_line_name=="all":
        drug_comb=extract
    else:
        if type(cell_line_name) is list:
            drug_comb=extract.loc[extract["cell_line_name"].isin(cell_line_name)]
        else:
            drug_comb=extract.loc[extract["cell_line_name"]==cell_line_name]
    
    n_sample=drug_comb.shape[0]
    n_feature=((phychem.shape[1]-1)+(finger.shape[1]-1)+978)*2+1+978
    drug_comb.index=range(n_sample)
    data=np.zeros((n_sample,n_feature))
    for i in range(n_sample):
        drugA_id=drug_comb.at[i,"drug_row_cid"]
        drugB_id=drug_comb.at[i,"drug_col_cid"]
        drugA_finger=get_finger(finger,drugA_id)
        drugB_finger=get_finger(finger,drugB_id)
        drugA_phychem=get_phychem(phychem,drugA_id)
        drugB_phychem=get_phychem(phychem,drugB_id)
        cell_line_name=drug_comb.at[i,"cell_line_name"]
        drugA_express=get_express(all_express[cell_line_name],drugA_id)
        drugB_express=get_express(all_express[cell_line_name],drugB_id)
        feature=get_cell_feature(cell_feature,cell_line_name)
        label=drug_comb.at[i,"S"]
        sample=np.hstack((drugA_finger,drugA_phychem,drugA_express,drugB_finger,drugB_phychem,drugB_express,feature,label))
        data[i]=sample
    return data


def get_finger(finger,drug_id):
    drug_finger=finger.loc[finger['drug_id']==drug_id]
    drug_finger=np.array(drug_finger)
    drug_finger=drug_finger.reshape(drug_finger.shape[1])[1:]
    return drug_finger

def get_phychem(phychem,drug_id):
    drug_phychem=phychem.loc[phychem["cid"]==drug_id]
    drug_phychem=np.array(drug_phychem)
    drug_phychem=drug_phychem.reshape(drug_phychem.shape[1])[1:]
    return drug_phychem

def get_express(express,drug_id):
    drug_express=express[str(drug_id)]
    drug_express=np.array(drug_express)
    return drug_express

def get_cell_feature(feature,cell_line_name):
    cell_feature=feature[str(cell_line_name)]
    cell_feature=np.array(cell_feature)
    return cell_feature

