#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Created on Thu Nov 03 09:28:09 2016
@author: Sidwell Rigade
Description:
	Gaussian mixture with an exptezation maximisation algorithm to genotype
	snp in 3 genotypes.

Usage:
    GaussianMixture.py --snpList=File --callingRAC=File

Options:
    -h --help			Show help
    -V --version		Show the version and exit
    --snpList			list snp txt
    --callingRAC		results snp

"""
__version__ = '0.0.1'

try:
    import docopt
except ImportError:
    print("package docopt needed, use this cmd:\n pip install "+ "docopt")
    rapport.write("package docopt needed, use this cmd:\npip install "+ "docopt")
    exit()
import numpy as np # Numpy v1.11.2
import matplotlib.pyplot as plt # matplotlib v1.5.3
import matplotlib as mpl
import pandas as pd # pandas v0.19.1
from sklearn import mixture # sklearn v0.18
import scipy # scipy v0.18.1
from scipy import linalg
import time 
import itertools
import progressbar
from time import sleep
from hmmlearn import hmm
import math
from collections import Counter
pd.set_option('chained',None) #pour eviter le warning sur les copies de dataframe


################ IMPORT DATAS #################
def importDatas(fileToImport): # import file in dataframe and add header, chose wich one 
    # import datas as dataframe with pandas
    df = pd.read_csv(fileToImport, sep='\t')
    df.columns = ['Plate','snp','Sample','ProbeSetName','Call','Confidence','Log Ratio','Strength','Forced Call']
    return df
def degPlatesFromSNP(df,name_snp): # split dataframe by snp 
    table_snp = df.loc[df.snp==name_snp]
    return table_snp
def import_list_snp(listSNP): # import a list with snp names like "rs...." (file.txt) 
    list_snp = []
    with open(listSNP,'r') as f:
        for snp in f.readlines():
            snp = snp.replace("\n","")
            list_snp.append(snp)
    return list_snp 
def getMAF(fileAllMAF,name_snp): # get the MAF from file created by "getMAF.py" -> get MAF from annote files 
    dfMAF = pd.read_csv(fileAllMAF)
    MAFsnp = dfMAF.loc[dfMAF.snp==name_snp]['maf'].item()
    alleleMAF = dfMAF.loc[dfMAF.snp==name_snp]['allele'].item()
    MAF_allele = [MAFsnp,alleleMAF]
    return MAF_allele 

############ PARAMETERS TO DO GAUSSIAN MIXTURE #############
def freqGenotypes(maf):
    maf = float(maf[0])
    freqBB = maf**2
    freqAB = 2 * maf * (1 - maf)
    freqAA = (1-maf)**2
    weigths_theo = np.asarray([freqAA,freqAB,freqBB])
    return weigths_theo
def getADN103(df): # recupere les ADN103 pour les colorer ensuite sur le graph 
    expreg = "ADN103_RAC."
    ADN_103 = df[df['Sample'].str.contains(expreg, regex=True)]
    return ADN_103
def getLocADN103(df):
    return df.index.get_loc(getADN103(df).index[0])
def scoreHomogeneite(dico_ADN103):
    counting = Counter(dico_ADN103.values())
    scoreHomo = max(counting.values())/len(dico_ADN103)
    return scoreHomo
def genoMaxADN103(dico_ADN103):
    return max(Counter(dico_ADN103.values()))
def definePriors(df,maf,rapport): # define priors from maf to re genotype after 
    N = len(df)
    maf_num = maf[0]
    nbAB = 2 * N * maf_num * (1 - maf_num)
    nbAB = int(nbAB)
    a = int(nbAB/2)
    if maf[1] == 'A':
        nbAA = N * (maf_num**2)
        nbAA = int(nbAA)
        nbBB = N * ((1-maf_num)**2)
        nbBB = int(nbBB)
        df = df.sort_values(["Log Ratio"],ascending=True) # small values on top
        new_priorBB_log = df.iloc[int((0+nbBB)/2)]["Log Ratio"]
        priorBB_strength = df.iloc[int((0+nbBB)/2)]["Strength"]
        new_priorAB_log = df.iloc[nbBB+a]["Log Ratio"]
        priorAB_strength = df.iloc[nbBB+a]["Strength"]
        b = int(nbAA/2)
        try:
            end_last_cluster = df.iloc[nbBB+nbAB+nbAA]["Log Ratio"]
        except IndexError:
            rapport.write('last theorical cluster is ouf of plate\n')
        new_priorAA_log = df.iloc[nbBB+nbAB+b]["Log Ratio"]
        priorAA_strength = df.iloc[nbBB+nbAB+b]["Strength"]
    if maf[1] == 'B':
        df = df.sort_values(["Log Ratio"],ascending=False) # high values on top
        nbBB = N * (maf_num**2)
        nbBB = int(nbBB)
        nbAA = N * ((1-maf_num)**2)
        nbAA = int(nbAA)
        new_priorAA_log = df.iloc[int((0+nbAA)/2)]["Log Ratio"] # parmi le nombre theorique de AA : prend le point avec un logRatio median
        priorAA_strength = df.iloc[int((0+nbAA)/2)]["Strength"]
        new_priorAB_log = df.iloc[nbAA+a]["Log Ratio"]
        priorAB_strength = df.iloc[nbAA+a]["Strength"]
        b = int(nbBB/2)
        try:
            end_last_cluster = df.iloc[nbAA+nbAB+nbBB]["Log Ratio"]
        except IndexError:
            rapport.write('last theorical cluster is ouf of plate\n')
        new_priorBB_log = df.iloc[nbAA+nbAB+b]["Log Ratio"] # parmi le nombre theorique de BB (on ajoute le nb de AA et de BB pour arriver a lindice)
        priorBB_strength = df.iloc[nbAA+nbAB+b]["Strength"]
    priors = [[new_priorAA_log,priorAA_strength],[new_priorAB_log,priorAB_strength],[new_priorBB_log,priorBB_strength]]
    priors = np.asarray(priors)
    return priors

############### GAUSSIAN MIXTURE #############
def GaussianMixture(df,maf,starting_points): # do the GM with EM algorithm and output results 
    datas = df[["Log Ratio", "Strength"]].as_matrix()
    gm = mixture.GaussianMixture(n_components=3, covariance_type="tied",weights_init=freqGenotypes(maf),means_init=starting_points,random_state=0, max_iter=200, verbose=0, warm_start=False).fit(datas)
    resultatsGM = gm.predict(datas)
    return resultatsGM 
def plot_results(df,resultGM): # plot results on graph 
    colors_genotypes = ['pink' if i==0 else 'skyblue' if i==1 else 'lightgreen' for i in resultGM]
    ax=plt.gca()
    df_to_plot = df[["Log Ratio", "Strength"]].as_matrix()
    ax.scatter(df_to_plot[:,0], df_to_plot[:,1], alpha=0.8, c=colors_genotypes)
    ax.scatter(df_to_plot[getLocADN103(df),0], df_to_plot[getLocADN103(df),1], alpha=1, c="black")
    return ax
def addLegend_on_graph(ax,name_snp,maf,score,HW,maf_obs,effectifs_obs,step):
    plt.title("Calling "+name_snp)
    plt.xlabel('LogRatio')
    plt.ylabel('Strength')
    plt.text(0.015, 0.98,"Score homogeneite = "+str(score),fontsize=8, ha='left', va='center', transform=ax.transAxes)
    if step == 0:
        plt.text(0.015, 0.95,"MAF_theo = "+str(maf[0])+"  allele:"+str(maf[1]),fontsize=8, ha='left', va='center', transform=ax.transAxes)
    if step == 1:
        plt.text(0.015, 0.95,"MAF_obs_step1 = "+str(maf[0])+"  allele:"+str(maf[1]),fontsize=8, ha='left', va='center', transform=ax.transAxes)
    plt.text(0.015, 0.92,"HW = "+str(HW),fontsize=8, ha='left', va='center', transform=ax.transAxes)
    plt.text(0.015, 0.89,"MAF_obs = "+str(maf_obs[0])+"  allele:"+str(maf_obs[1]),fontsize=8, ha='left', va='center', transform=ax.transAxes)
    plt.text(0.015, 0.86,"NbAA:"+str(effectifs_obs[0])+"  nbAB:"+str(effectifs_obs[1])+"  nbBB:"+str(effectifs_obs[2]),fontsize=8, ha='left', va='center', transform=ax.transAxes)
    if step == 0:
        plt.savefig(str(name_snp)+"_"+str(step)+"calling_old_priors.png")
    if step == 1:
        plt.savefig(str(name_snp)+"_"+str(step)+"calling_new_priors.png")
    plt.clf()

############## RESULTS ################
def clusterTOgeno(point):
    if point == 0:
        return "AA"
    if point == 1:
        return "AB"
    if point == 2:
        return "BB"
def effectifs_obs(results): # get observed effectives of each genotype after GM by prediction 
    nbAA = results.count(0)
    nbAB = results.count(1)
    nbBB = results.count(2)
    effectifsAfterGM = [nbAA,nbAB,nbBB]
    return effectifsAfterGM
def effectifs_theo(N,maf):
    maf = maf[0]
    theo_eff_homo_variant = N * (maf**2)
    theo_eff_AB = 2 * N * maf * (1 - maf)
    theo_eff_homo_ref = N * ((1-maf)**2)
    theo_effectifs = [theo_eff_homo_ref, theo_eff_AB, theo_eff_homo_variant]
    return theo_effectifs
def getMAFobs(effectifs_obs): # calcul MAF observed with effectifs predicted 
    freq_A = ((2 * effectifs_obs[0]) + effectifs_obs[1]) / (2 * (sum(effectifs_obs)))
    freq_B = ((2 * effectifs_obs[2]) + effectifs_obs[1]) / (2 * (sum(effectifs_obs)))
    if freq_A < freq_B:
        mafOBS = [freq_A,'A']
    else :
        mafOBS = [freq_B,'B']
    return mafOBS # return the smaller frequence = MAF
def hardyWeinberg(eff_obs,maf_obs,eff_theo): # return a chi2 
    chi2 = (((eff_theo[0]-eff_obs[0])**2)/eff_obs[0])+(((eff_theo[1]-eff_obs[1])**2)/eff_obs[1])+(((eff_theo[2]-eff_obs[2])**2)/eff_obs[2])
    pval = (scipy.stats.chisquare(eff_obs, f_exp=eff_theo, ddof=1)[1])
    return pval
def priors_center_on_ADN103(name_plate,dico_ADN103,mafOBS,df,resultsGM,rapport):
    N = len(df)
    maf = mafOBS[0]
    nbAB = 2 * N * maf * (1 - maf)
    nbAB = int(nbAB)
    demiAB = int(nbAB/2)
    if mafOBS[1] == 'A':
        df = df.sort_values(["Log Ratio"],ascending=True) # small values on top
        ADN103 = getLocADN103(df)
        nbAA = N * (maf**2)
        nbAA = int(nbAA)
        demiAA = int(nbAA/2)
        nbBB = N * ((1-maf)**2)
        nbBB = int(nbBB)
        demiBB = int(nbBB/2)
        if genoMaxADN103(dico_ADN103) == 'BB':
            new_priorBB_log = df.iloc[ADN103]["Log Ratio"]
            priorBB_strength = df.iloc[ADN103]["Strength"]
            if ADN103+demiBB+demiAB < len(df):
                new_priorAB_log = df.iloc[int(ADN103+demiBB+demiAB)]["Log Ratio"]
                priorAB_strength = df.iloc[int(ADN103+demiBB+demiAB)]["Strength"]
            else:
                rapport.write('ADN103 on %rplate is %r position of %r -> new priors ouf of bounds\n' % (name_plate,ADN103,len(df)))
                return definePriors(df,mafOBS,rapport)
            new_priorAA_log = df.iloc[int(ADN103+demiBB+nbAB+demiAA)]["Log Ratio"]
            priorAA_strength = df.iloc[int(ADN103+demiBB+nbAB+demiAA)]["Strength"]
        if genoMaxADN103(dico_ADN103) == 'AA':
            new_priorAA_log = df.iloc[ADN103]["Log Ratio"]
            priorAA_strength = df.iloc[ADN103]["Strength"]
            new_priorAB_log = df.iloc[int(ADN103-demiAA-demiAB)]["Log Ratio"]
            priorAB_strength = df.iloc[int(ADN103-demiAA-demiAB)]["Strength"]
            new_priorBB_log = df.iloc[int(ADN103-demiAA-nbAB-demiBB)]["Log Ratio"]
            priorBB_strength = df.iloc[int(ADN103-demiAA-nbAB-demiBB)]["Strength"]
        if genoMaxADN103(dico_ADN103) == 'AB':
            new_priorAB_log = df.iloc[ADN103]["Log Ratio"]
            priorAB_strength = df.iloc[ADN103]["Strength"]
            new_priorBB_log = df.iloc[int(ADN103-demiAB-demiBB)]["Log Ratio"]
            priorBB_strength = df.iloc[int(ADN103-demiAB-demiBB)]["Strength"]
            if ADN103+demiAB+demiAA < len(df):
                new_priorAA_log = df.iloc[int(ADN103+demiAB+demiAA)]["Log Ratio"]
                priorAA_strength = df.iloc[int(ADN103+demiAB+demiAA)]["Strength"]
            else:
                rapport.write('ADN103 on %rplate is %r position of %r -> new priors ouf of bounds\n' % (name_plate,ADN103,len(df)))
                return definePriors(df,mafOBS,rapport)
    if mafOBS[1] == 'B':
        df = df.sort_values(["Log Ratio"],ascending=False) # high values on top
        ADN103 = getLocADN103(df)
        nbBB = N * (maf**2)
        nbBB = int(nbBB)
        demiBB = int(nbBB/2)
        nbAA = N * ((1-maf)**2)
        nbAA = int(nbAA)
        demiAA = int(nbAA/2)
        if genoMaxADN103(dico_ADN103) == 'AA':
            new_priorAA_log = df.iloc[ADN103]["Log Ratio"]
            priorAA_strength = df.iloc[ADN103]["Strength"]
            new_priorAB_log = df.iloc[int(ADN103+demiAA+demiAB)]["Log Ratio"]
            priorAB_strength = df.iloc[int(ADN103+demiAA+demiAB)]["Strength"]
            new_priorBB_log = df.iloc[int(ADN103+demiAA+nbAB+demiBB)]["Log Ratio"]
            priorBB_strength = df.iloc[int(ADN103+demiAA+nbAB+demiBB)]["Strength"]
        if genoMaxADN103(dico_ADN103) == 'BB':
            new_priorBB_log = df.iloc[ADN103]["Log Ratio"]
            priorBB_strength = df.iloc[ADN103]["Strength"]
            new_priorAB_log = df.iloc[int(ADN103-demiBB-demiAB)]["Log Ratio"]
            priorAB_strength = df.iloc[int(ADN103-demiBB-demiAB)]["Strength"]
            new_priorAA_log = df.iloc[int(ADN103-demiBB-nbAB-demiAA)]["Log Ratio"]
            priorAA_strength = df.iloc[int(ADN103-demiBB-nbAB-demiAA)]["Strength"]
        if genoMaxADN103(dico_ADN103) == 'AB':
            new_priorAB_log = df.iloc[ADN103]["Log Ratio"]
            priorAB_strength = df.iloc[ADN103]["Strength"]
            new_priorBB_log = df.iloc[int(ADN103+demiAB+demiBB)]["Log Ratio"]
            priorBB_strength = df.iloc[int(ADN103+demiAB+demiBB)]["Strength"]
            new_priorAA_log = df.iloc[int(ADN103-demiAB-demiAA)]["Log Ratio"]
            priorAA_strength = df.iloc[int(ADN103-demiAB-demiAA)]["Strength"]
    priors = [[new_priorAA_log,priorAA_strength],[new_priorAB_log,priorAB_strength],[new_priorBB_log,priorBB_strength]]
    priors = np.asarray(priors)
    return priors

################## MAIN ################
def main():
    
    genotypes = ["AA","AB","BB"]
    AllDatas = importDatas(args['--callingRAC'])
    AllDatas = AllDatas.sort_values(['snp','Sample'], ascending=[True,False]) #sort by snp to keep indices to plot after
    
    listSNP = import_list_snp(args['--snpList'])

    #listSNP = ['rs35386136']
    #listSNP = ['rs34014260']
    #listSNP = ['rs6639113']
    #listSNP = ['rs2212606']
    #listSNP = ['rs34440822']
    #listSNP=['rs735476']
    #listSNP=['rs9908342']

    rapport = open("rapport_galwanII.txt",'w')

    for snp in listSNP:
        print(snp)
        df = degPlatesFromSNP(AllDatas,snp)
        try:
            maf_theo = getMAF("list_mafs.txt",snp)
        except FileNotFoundError:
            print("Run getMAF.py first to get MAF from annotation file")
            break
                
        rapport.write('SNP: %r\tmaf: %r\n\n' % (snp,maf_theo))
        print("SNP:",snp, "\tmaf:",maf_theo)

        dico_priors = {}
        dicoADN103 = {}
        allplates_resultsGM_step1 = []
        print("STEP 1: define priors with theorical MAF")
        rapport.write("STEP 1: define priors with theorical MAF\n")

        for name, group in df.groupby('Plate'):
            print("Calling Plate",name)
            dico_priors[name] = definePriors(group,maf_theo,rapport)
            rapport.write('Plate%r: Priors = AA:%.2f/%.2f\tAB:%.2f/%.2f\tBB:%.2f/%.2f --> Calling Plate\n' % (name,dico_priors[name][0][0],dico_priors[name][0][1],dico_priors[name][1][0],dico_priors[name][1][1],dico_priors[name][2][0],dico_priors[name][2][1]))

            resultGM_step1 = GaussianMixture(group,maf_theo,dico_priors[name])

            for indiv_cluster in resultGM_step1:
                allplates_resultsGM_step1.append(indiv_cluster)
            plot_step1 = plot_results(group,resultGM_step1)
            dicoADN103[name] = clusterTOgeno(resultGM_step1[getLocADN103(group)])

        eff_obs_step1 = effectifs_obs(allplates_resultsGM_step1)
        maf_obs_step1 = getMAFobs(eff_obs_step1)
        eff_theo_step1 = effectifs_theo(len(df),maf_obs_step1)
        HW_step1 = hardyWeinberg(eff_obs_step1,maf_obs_step1,eff_theo_step1)
        score_step1 = scoreHomogeneite(dicoADN103)

        addLegend_on_graph(plot_step1,snp,maf_theo,score_step1,HW_step1,maf_obs_step1,eff_obs_step1,0) 

        rapport.write("\n")
        print("ADN103 call in all plates after 1st step of calling")
        rapport.write("ADN103 call in all plates after 1st step of calling = %r\n" % (dicoADN103))
        print(dicoADN103)
        rapport.write("Score homogeneite step1 : %r | %r\n" % (score_step1,genoMaxADN103(dicoADN103)))
        rapport.write("Effectifs observed step1 : AA:%r|AB:%r|BB:%r\n" % (eff_obs_step1[0],eff_obs_step1[1],eff_obs_step1[2]))
        print("Observed MAF after step1 : ",maf_obs_step1)
        rapport.write('Observed MAF after step1 : %r\n\n' % (maf_obs_step1))

        dico_new_priors = {}
        new_dicoADN103 = {}
        allplates_resultsGM_step2 = []
        print("STEP 2: redefine priors with observed maf and center on ADN103 if different from other plates")
        rapport.write("STEP 2: redefine priors with observed maf and center on ADN103 if different from other plates\n")

        for name, group in df.groupby('Plate'):
            print("Calling Plate",name)

            if dicoADN103[name] == genoMaxADN103(dicoADN103):
                dico_new_priors[name] = definePriors(group,maf_obs_step1,rapport)
                rapport.write('Plate%r: ADN103 ok -> Priors with OBS MAF = AA:%.2f/%.2f\tAB:%.2f/%.2f\tBB:%.2f/%.2f --> Calling Plate\n' % (name,dico_priors[name][0][0],dico_priors[name][0][1],dico_priors[name][1][0],dico_priors[name][1][1],dico_priors[name][2][0],dico_priors[name][2][1]))
            if dicoADN103[name] != genoMaxADN103(dicoADN103):
                print("ADN 103 is different from the other plates")
                dico_new_priors[name] = priors_center_on_ADN103(name,dicoADN103,maf_obs_step1,group,allplates_resultsGM_step1,rapport)
                rapport.write('Plate%r: ADN 103 is different from the other plates -> Priors center on ADN103 = AA:%.2f/%.2f\tAB:%.2f/%.2f\tBB:%.2f/%.2f --> Calling Plate\n' % (name,dico_priors[name][0][0],dico_priors[name][0][1],dico_priors[name][1][0],dico_priors[name][1][1],dico_priors[name][2][0],dico_priors[name][2][1]))
            

            resultsGM_step2 = GaussianMixture(group,maf_obs_step1,dico_new_priors[name])

            for indiv_cluster in resultsGM_step2:
                allplates_resultsGM_step2.append(indiv_cluster)
            plot_step2 = plot_results(group,resultsGM_step2) 
            new_dicoADN103[name] = clusterTOgeno(resultsGM_step2[getLocADN103(group)])

        eff_obs_step2 = effectifs_obs(allplates_resultsGM_step2)
        maf_obs_step2 = getMAFobs(eff_obs_step2)
        eff_theo_step2 = effectifs_theo(len(df),maf_obs_step2)
        HW_step2 = hardyWeinberg(eff_obs_step2,maf_obs_step2,eff_theo_step2)
        score_step2 = scoreHomogeneite(new_dicoADN103)

        addLegend_on_graph(plot_step2,snp,maf_obs_step1,score_step2,HW_step2,maf_obs_step2,eff_obs_step2,1)

        rapport.write("\n")
        rapport.write("ADN103 call in all plates after 2nd step of calling = %r\n" % (new_dicoADN103))
        rapport.write("Score homogeneite step2 : %r | %r\n" % (score_step2,genoMaxADN103(new_dicoADN103)))
        rapport.write("Effectifs observed step2 : AA:%r|AB:%r|BB:%r\n" % (eff_obs_step2[0],eff_obs_step2[1],eff_obs_step2[2]))
        print("Observed MAF after step2 : ",maf_obs_step2)
        rapport.write('Observed MAF after step2 : %r\n\n' % (maf_obs_step2))

        print("Calling for all plates of this snp done\n\n\n")
        rapport.write("Calling for all plates done -> add to the plot\n\n\n")
        rapport.write("#################################################################\n")

    print("Calling for all snps done and rapport created")
    rapport.write("Calling for all snps done and rapport created")
    

if __name__ == '__main__' :
    args = docopt.docopt(__doc__, version=__version__)
    main()