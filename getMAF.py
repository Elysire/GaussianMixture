#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Created on Thu Nov 03 09:28:09 2016
@author: Sidwell Rigade
Description:
	Get MAF from anotation file or from an adding file with missing MAF from dbSNP

Usage:
    getMAF.py --listSNP=file --annotFile=File --missingMAF=File

Options:
    -h --help       Show help.
    listSNP         list of SNP to genotype
    annotFile       annotation file
    missingMAF      MAF that are not in annote file

"""

try:
    import docopt
except ImportError:
    print("package docopt needed, use this cmd:\n pip install "+ "docopt")
    exit()
import pandas as pd # pandas v0.19.1
from itertools import chain
pd.set_option('chained',None) #pour eviter le warning sur les copies de dataframe

def import_list_snp(listSNP):
    list_snp = []
    with open(listSNP,'r') as f:
        for snp in f.readlines():
            snp = snp.replace("\n","")
            list_snp.append(snp)
    return list_snp

def getMAF_from_annot_file(filepath,name_snp): # get MAF for each snp 
    df = pd.read_csv(filepath,sep=',',header=19,low_memory=False)
    if df[df['dbSNP RS ID']==name_snp]['Minor Allele Frequency'].any():
        line = df[df['dbSNP RS ID']==name_snp]['Minor Allele Frequency']
        line = line.str.split("/")
        allele_frequency=line.iloc[0][0]
        return allele_frequency
    else:
        allele_frequency="snp not in anotation file"
    return allele_frequency

def getMAFfromMAF_file(MAFfile):
    with open(MAFfile,'r') as f:
        list_maf_snp=[]
        for line in f.readlines():
            couple_snp_maf=[]
            line = line.replace("\n","")
            line = line.split("\t")
            name_snp = line[0]
            maf = line[1]
            couple_snp_maf.append(name_snp)
            couple_snp_maf.append(maf)
            list_maf_snp.append(couple_snp_maf)
        return list_maf_snp
    
def main():
    args = docopt.docopt(__doc__)
    tablo_all_maf=[]
    listSNP = import_list_snp(args['--listSNP'])
    missing_maf_file = getMAFfromMAF_file(args['--missingMAF'])
    for snp in listSNP:
        if snp in chain(*missing_maf_file):
            for i in missing_maf_file:
                if i[0] == snp:
                    maf = i[1]
        if snp not in chain(*missing_maf_file):
            maf = getMAF_from_annot_file(args['--annotFile'],snp)
        if maf == "snp not in anotation file" or maf == "---":
            print(snp,"can't get maf")
            continue
        couple_snp_maf_final = [snp,maf]
        print(couple_snp_maf_final)
        tablo_all_maf.append(couple_snp_maf_final)
    all_maf = pd.DataFrame(tablo_all_maf,columns=['snp','maf'])
    print(all_maf)
    all_maf.to_csv(path_or_buf="all_maf_4_GM.txt")
    return all_maf
    

if __name__ == '__main__' :
    main()

