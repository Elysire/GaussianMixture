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
pd.set_option('chained',None) # pour eviter le warning sur les copies de dataframe

def import_list_snp(listSNP): # list of snp names
    list_snp = []
    with open(listSNP,'r') as f:
        for snp in f.readlines():
            snp = snp.replace("\n","")
            list_snp.append(snp)
    return list_snp

def getMAF_from_annot_file34(filepath,name_snp): # get MAF for each snp 
    df = pd.read_csv(filepath,sep=',',header=19,low_memory=False)
    if df[df['dbSNP RS ID']==name_snp]['Allele Frequencies'].any(): # if snp is this annot file
        lineFreq = df[df['dbSNP RS ID']==name_snp]['Allele Frequencies'] # get the line with the snp at the allele frequencies column
        if lineFreq.item() != "---": # if the frequence is not empty
            lineFreq = lineFreq.str.split("//")
            freq_A = lineFreq.item()[0]
            freq_A = float(freq_A)
            freq_B = lineFreq.item()[1]
            freq_B = float(freq_B)
            print("A:",freq_A, "B:",freq_B)
            if freq_B < freq_A :
                print("B smaller than A")
                maf = [freq_B,"B"]
            if freq_A < freq_B:
                print("A smaller than B")
                maf = [freq_A,"A"]
            print("SO maf = ",maf)
            return maf
        else:
            maf=lineFreq.item() # the frequence is empty
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
        freq_A = float(freq_A)
        freq_B = lineFreq.item()[4]
        freq_B = float(freq_B)
        print("A:",freq_A, "B:",freq_B)
        if freq_B < freq_A :
            print("B smaller than A")
            maf = [freq_B,"B"]
        if freq_A < freq_B:
            print("A smaller than B")
            maf = [freq_A,"A"]
        print("SO maf =",maf)
    return maf

    
def main():
    args = docopt.docopt(__doc__)
    tablo_all_maf=[]
    listSNP = import_list_snp(args['--snpList'])
    #listSNP=['rs2487622']
    for snp in listSNP:
        maf = getMAF_from_annot_file34(args['--annotFile2'],snp)
        if maf != "---":
            print("ANNOT34")
        if maf == "---": # if maf is not in annot34, it must be in annot32
            print("ANNOT32")
            maf = getMAF_from_annot_file32(args['--annotFile1'],snp)
        couple_snp_maf_final = [snp,maf[0],maf[1]]
        print(couple_snp_maf_final)
        tablo_all_maf.append(couple_snp_maf_final)
    all_maf = pd.DataFrame(tablo_all_maf,columns=['snp','maf','allele'])
    print(all_maf)
    all_maf.to_csv(path_or_buf="list_mafs.txt") # return maf in a file to read it from an other script
    return all_maf
    

if __name__ == '__main__' :
    main()

