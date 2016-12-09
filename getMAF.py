#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Created on Thu Nov 03 09:28:09 2016
@author: Sidwell Rigade
Description:
	Get MAF from anotation file32 or 34

Usage:
    getMAF.py --snpList=file --annotFile1=File --annotFile2=File

Options:
    -h --help       Show help.
    snpList         list of SNP to genotype
    annotFile1      annotation file32
    annotFile2      Anntotation file34

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

def getMAF_from_annot_file34(filepath,name_snp): # get MAF for each snp 
    df = pd.read_csv(filepath,sep=',',header=19,low_memory=False)
    if df[df['dbSNP RS ID']==name_snp]['Allele Frequencies'].any():
        lineFreq = df[df['dbSNP RS ID']==name_snp]['Allele Frequencies']
        if lineFreq.item() != "---":
            lineFreq = lineFreq.str.split("//")
            freq_A = lineFreq.item()[0]
            freq_B = lineFreq.item()[1]
            if freq_B < freq_A :
                maf = [freq_B,"B"]
            if freq_A < freq_B:
                maf = [freq_A,"A"]
            return maf
        else:
            maf=lineFreq.item()
    else :
        maf = "---"
    return maf

def getMAF_from_annot_file32(filepath,name_snp):
    df = pd.read_csv(filepath,sep=',',header=20,low_memory=False)
    if df[df['dbSNP RS ID']==name_snp]['Allele Frequencies'].any():
        lineFreq = df[df['dbSNP RS ID']==name_snp]['Allele Frequencies']
        lineFreq = lineFreq.str.split("//")
        freq_A = lineFreq.item()[3]
        freq_A = freq_A.replace("/ ","")
        freq_B = lineFreq.item()[4]
        if freq_B < freq_A :
            maf = [freq_B,"B"]
        if freq_A < freq_B:
            maf = [freq_A,"A"]
    return maf

    
def main():
    args = docopt.docopt(__doc__)
    tablo_all_maf=[]
    listSNP = import_list_snp(args['--snpList'])
    #listSNP=['rs34440822']
    for snp in listSNP:
        maf = getMAF_from_annot_file34(args['--annotFile2'],snp)
        if maf == "---":
            maf = getMAF_from_annot_file32(args['--annotFile1'],snp)
        couple_snp_maf_final = [snp,maf[0],maf[1]]
        print(couple_snp_maf_final)
        tablo_all_maf.append(couple_snp_maf_final)
    all_maf = pd.DataFrame(tablo_all_maf,columns=['snp','maf','allele'])
    print(all_maf)
    all_maf.to_csv(path_or_buf="all_maf_4_GM.txt")
    return all_maf
    

if __name__ == '__main__' :
    main()

