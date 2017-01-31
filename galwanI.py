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

############## CORRECTION DATAS #############
def quartile_subset(logratios,lower,upper): # avoid outliers 
    #subset logRatios to not count extremes in mean
    #logratio v must be higher than the lower quantile and lower than the upper quantile to avoid outliers
    return logratios.loc[[True if v < logratios.quantile(q=upper) and v > logratios.quantile(q=lower) else False for v in logratios]]
def dist_inter_quantile(serie,lower,upper): # distance between two quantiles 
    dist = serie.quantile(q=upper) - serie.quantile(q=lower)
    return dist
def correct_strength_median(df): # correct strength by median 
    df['medianStrengthPlate'] = df.groupby('Plate')['Strength'].transform('median')
    df['CorrectedStrength'] = df['Strength'] - df.groupby('Plate')['Strength'].transform('median')
    return df
def correct_centrality_entendue_with_ADN_103(df):
    df['m1'] = df.groupby('Plate')['Log Ratio'].transform(lambda s: quartile_subset(s,0.20,0.80).mean())
    df['m2'] = df.groupby('Plate')['Log Ratio'].transform(lambda s: quartile_subset(s,0.25,0.75).mean())
    df['m3'] = df.groupby('Plate')['Log Ratio'].transform(lambda s: quartile_subset(s,0.30,0.70).mean())
    df['Ei'] = df.groupby('Plate')['Log Ratio'].transform(lambda i: (dist_inter_quantile(i,0.20,0.80)+dist_inter_quantile(i,0.25,0.75)+dist_inter_quantile(i,0.30,0.70))/3)
    maxEI = df['Ei'].max()
    df.set_index('Plate', inplace=True)
    for plate in df.index.unique():
        indx = np.where(df.loc[plate, 'Sample'].str.contains('ADN'))[0][0]
        temp_value = df.loc[plate, 'Log Ratio'].iat[indx]
        df.loc[plate, 'ADN_LogValues'] = temp_value
    df['Mold'] = (df['m1'] + df['m2'] + df['m3'])/3
    df['M'] = (df['m1'] + df['m2'] + df['m3'] + df['ADN_LogValues'])/4
    df['CorrectedLogRatio_centrality'] = df['Log Ratio'] - df['M']
    df.reset_index(level=0,inplace=True)
    df['CorrectedLogRatio'] = df['CorrectedLogRatio_centrality'] * df.groupby('Plate')['Ei'].transform(lambda x : maxEI/x)
    df.to_csv(path_or_buf="/home/e062531t/Bureau/Sidwell/res/corrected.txt")
    return df

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
def listIndexADN103(df):
    return getADN103(df).index.tolist()
def scoreHomogeneite(df,results):
    nbAA=0
    nbAB=0
    nbBB=0
    for i in listIndexADN103(df):
        if clusterTOgeno(results[i]) == 'AA':
            nbAA+=1
        if clusterTOgeno(results[i]) == 'AB':
            nbAB+=1
        if clusterTOgeno(results[i]) == 'BB':
            nbBB+=1
    maxGeno = max(nbAA,nbAB,nbBB)
    score = maxGeno / len(listIndexADN103(df))
    if maxGeno == nbAA:
        maxGenoADN103 = "AA"
    if maxGeno == nbAB:
        maxGenoADN103 = "AB"
    if maxGeno == nbBB:
        maxGenoADN103 = "BB"
    return score, maxGenoADN103
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
        df = df.sort_values(["CorrectedLogRatio"],ascending=True) # small values on top
        new_priorBB_log = df.iloc[int((0+nbBB)/2)]["CorrectedLogRatio"]
        priorBB_strength = df.iloc[int((0+nbBB)/2)]["CorrectedStrength"]
        new_priorAB_log = df.iloc[nbBB+a]["CorrectedLogRatio"]
        priorAB_strength = df.iloc[nbBB+a]["CorrectedStrength"]
        b = int(nbAA/2)
        try:
            end_last_cluster = df.iloc[nbBB+nbAB+nbAA]["CorrectedLogRatio"]
        except IndexError:
            rapport.write('last theorical cluster is ouf of plate\n')
        new_priorAA_log = df.iloc[nbBB+nbAB+b]["CorrectedLogRatio"]
        priorAA_strength = df.iloc[nbBB+nbAB+b]["CorrectedStrength"]
    if maf[1] == 'B':
        df = df.sort_values(["CorrectedLogRatio"],ascending=False) # high values on top
        nbBB = N * (maf_num**2)
        nbBB = int(nbBB)
        nbAA = N * ((1-maf_num)**2)
        nbAA = int(nbAA)
        new_priorAA_log = df.iloc[int((0+nbAA)/2)]["CorrectedLogRatio"] # parmi le nombre theorique de AA : prend le point avec un logRatio median
        priorAA_strength = df.iloc[int((0+nbAA)/2)]["CorrectedStrength"]
        new_priorAB_log = df.iloc[nbAA+a]["CorrectedLogRatio"]
        priorAB_strength = df.iloc[nbAA+a]["CorrectedStrength"]
        b = int(nbBB/2)
        try:
            end_last_cluster = df.iloc[nbAA+nbAB+nbBB]["CorrectedLogRatio"]
        except IndexError:
            rapport.write('last theorical cluster is ouf of plate\n')
        new_priorBB_log = df.iloc[nbAA+nbAB+b]["CorrectedLogRatio"] # parmi le nombre theorique de BB (on ajoute le nb de AA et de BB pour arriver a lindice)
        priorBB_strength = df.iloc[nbAA+nbAB+b]["CorrectedStrength"]
    priors = [[new_priorAA_log,priorAA_strength],[new_priorAB_log,priorAB_strength],[new_priorBB_log,priorBB_strength]]
    priors = np.asarray(priors)
    return priors

############### GAUSSIAN MIXTURE #############
def GaussianMixture(df,maf,starting_points): # do the GM with EM algorithm and output results 
    datas = df[["CorrectedLogRatio", "CorrectedStrength"]].as_matrix()
    gm = mixture.GaussianMixture(n_components=3, covariance_type="tied",weights_init=freqGenotypes(maf),means_init=starting_points,random_state=0, max_iter=200, verbose=0, warm_start=False).fit(datas)
    resultatsGM = gm.predict(datas)
    return resultatsGM 
def plot_results(df,resultGM): # plot results on graph 
    colors_genotypes = ['pink' if i==0 else 'skyblue' if i==1 else 'lightgreen' for i in resultGM]
    ax=plt.gca()
    df_to_plot = df[["CorrectedLogRatio", "CorrectedStrength"]].as_matrix()
    ax.scatter(df_to_plot[:,0], df_to_plot[:,1], alpha=0.8, c=colors_genotypes)
    for i in listIndexADN103(df):
        ax.scatter(df_to_plot[i,0], df_to_plot[i,1], alpha=1, c="black")
    return ax
def addLegend_on_graph(ax,name_snp,maf,score,HW,maf_obs,effectifs_obs,step):
    plt.title("Calling "+name_snp)
    plt.xlabel('LogRatio')
    plt.ylabel('CorrectedStrength')
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
    nbAA =  np.sum(results == 0)
    nbAB =  np.sum(results == 1)
    nbBB =  np.sum(results == 2)
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

    rapport = open("rapport_galwanI.txt",'w')

    for snp in listSNP:
        print(snp)
        df = degPlatesFromSNP(AllDatas,snp)
        try:
            maf_theo = getMAF("list_mafs.txt",snp)
        except FileNotFoundError:
            print("Run getMAF.py first to get MAF from annotation file")
            break
        
        print(effectifs_theo(len(df),maf_theo))
        
        rapport.write('SNP: %r\tmaf: %r\n\n' % (snp,maf_theo))
        print("SNP:",snp, "\tmaf:",maf_theo)

        print("Define priors with theorical MAF")
        rapport.write("Define priors with theorical MAF\n")

        df = correct_centrality_entendue_with_ADN_103(correct_strength_median(df))
        priors = definePriors(df,maf_theo,rapport)
        resultGM = GaussianMixture(df,maf_theo,priors)
        plot = plot_results(df,resultGM)

        eff_obs = effectifs_obs(resultGM)
        maf_obs = getMAFobs(eff_obs)
        eff_theo = effectifs_theo(len(df),maf_obs)
        HW = hardyWeinberg(eff_obs,maf_obs,eff_theo)
        score = scoreHomogeneite(df,resultGM)

        addLegend_on_graph(plot,snp,maf_theo,score[0],HW,maf_obs,eff_obs,0)

        rapport.write("\n")
        print("ADN103 call in all plates")
        rapport.write("Score homogeneite : %r | %r\n" % (score[0],score[1]))
        rapport.write("Effectifs observed : AA:%r|AB:%r|BB:%r\n" % (eff_obs[0],eff_obs[1],eff_obs[2]))
        print("Observed MAF : ",maf_obs)
        rapport.write('Observed MAF : %r\n\n' % (maf_obs))
        rapport.write("#################################################################\n")

    print("Calling for all snps done and rapport created")
    rapport.write("Calling for all snps done and rapport created")
    

if __name__ == '__main__' :
    args = docopt.docopt(__doc__, version=__version__)
    main()