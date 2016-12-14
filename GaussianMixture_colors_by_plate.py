#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Created on Thu Nov 03 09:28:09 2016
@author: Sidwell Rigade
Description:
	Gaussian mixture with an exptezation maximisation algorithm to genotype
	snp in 3 genotypes.

Usage:
    GaussianMixture.py --snpList=File --callingRAC=File --correction=<BOOL>

Options:
    -h --help			Show help
    -V --version		Show the version and exit
    --snpList			list snp txt
    --callingRAC		results snp
    --correction=BOOL	Perform correction ? True or False [default: True]  

"""
__version__ = '0.0.1'

try:
    import docopt
except ImportError:
    print("package docopt needed, use this cmd:\n pip install "+ "docopt")
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

############## HOMOGENEIZATION PLATES (CORRECTION) #############
def quartile_subset(logratios,lower,upper): # avoid outliers 
    #subset logRatios to not count extremes in mean
    #logratio v must be higher than the lower quantile and lower than the upper quantile to avoid outliers
    return logratios.loc[[True if v < logratios.quantile(q=upper) and v > logratios.quantile(q=lower) else False for v in logratios]]
def dist_inter_quantile(serie,lower,upper): # distance between two quantiles 
	dist = serie.quantile(q=upper) - serie.quantile(q=lower)
	return dist
def correct_logRatio_centrality_etendue(df): # correct log ratio by centrality and extended -> good correction with cool snp 
	df['m1'] = df.groupby('Plate')['Log Ratio'].transform(lambda s: quartile_subset(s,0.20,0.80).mean())
	df['m2'] = df.groupby('Plate')['Log Ratio'].transform(lambda s: quartile_subset(s,0.25,0.75).mean())
	df['m3'] = df.groupby('Plate')['Log Ratio'].transform(lambda s: quartile_subset(s,0.30,0.70).mean())
	
	"""
	expreg = "ADN103_RAC."
	ADN103 = df[df['Sample'].str.contains(expreg, regex=True)]
	logRatio_ADN103 = ADN103['Log Ratio']
	for name, group in df.groupby('Plate'):
		ADN103=group[group['Sample'].str.contains(expreg, regex=True)]
		df['ADN103'] = ADN103['Log Ratio']
		m = df['ADN103'].mean()
		for i in df['ADN103']:
			i=str(i)
			if i!="nan":
				value = i
		print(value)
		df['ADN103_bis']=""
		df['ADN103_bis'][df['Plate']==name] = value
		print(df['ADN103_bis'])
	df['ADN103_bis'] = df.groupby('Plate')[ADN103['Log Ratio']]
	print(df['ADN103_bis'])
	df['ADN103_bis'] = df.apply(lambda row: m[row['Plate']] if pd.isnull(row['ADN103']) else row['ADN103'])
	print(df['ADN103_bis'])
	"""

	df['M'] = (df['m1'] + df['m2'] + df['m3'])/3		
	df['CorrectedLogRatio_centrality'] = df['Log Ratio'] - df['M']
	df['Ei'] = df.groupby('Plate')['Log Ratio'].transform(lambda i: (dist_inter_quantile(i,0.20,0.80)+dist_inter_quantile(i,0.25,0.75)+dist_inter_quantile(i,0.30,0.70))/3)
	maxEI = df['Ei'].max()
	df['CorrectedLogRatio'] = df['CorrectedLogRatio_centrality'] * df.groupby('Plate')['Ei'].transform(lambda x : maxEI/x)
	df.to_csv(path_or_buf="/home/e062531t/Bureau/Sidwell/res/corrected.txt")
	return df
def correct_strength_median(df): # correct strength by median 
	df['medianStrengthPlate'] = df.groupby('Plate')['Strength'].transform('median')
	df['CorrectedStrength'] = df['Strength'] - df.groupby('Plate')['Strength'].transform('median')
	return df

############ PARAMETERS TO DO GAUSSIAN MIXTURE #############
def getParamsFromDatas(df_corrected, genotypes): # Get weigths for each genotype and means for each genotype for LogRatio and Strength -> choose datas corrected per plate or not 
	if args['--correction'] == 'True':
		LogRatio = "CorrectedLogRatio"
		Strength = "CorrectedStrength"
	else:
		LogRatio = "Log Ratio"
		Strength = "Strength"
	df = df_corrected
	df=df.rename(columns = {'Forced Call':'Forced_Call'}) #change name to be used in df.Forced_Call because spaces bugs
	wgts=[]
	means=[]
	countGeno=[]
	for i in genotypes :
		meansCouple=[] #create a temp couple Log-Stength to create a numpy array after
		nbGeno = df[df.Forced_Call==i].count()["Forced_Call"]
		countGeno.append(nbGeno)
		wgt = nbGeno/len(df) #count genotype i (AA or AB or BB) normalize with len
		wgts.append(wgt)
		meanLog = (df[df.Forced_Call==i].mean()[LogRatio]) #mean of logRatio for genotype i
		meansCouple.append(meanLog)
		meanStrength = (df[df.Forced_Call==i].mean()[Strength]) #mean of Strength for genotype i
		meansCouple.append(meanStrength)
		means.append(meansCouple)
		pvar=np.var(df[Strength])
	wgts = np.asarray(wgts)
	means_array = np.asarray(means)
	return wgts, means_array, countGeno, pvar
def getADN103(df): # recupere les indices des ADN103 par snp pour les colorer ensuite sur le graph 
	expreg = "ADN103_RAC."
	index_ADN_103 = df[df['Sample'].str.contains(expreg, regex=True)].index.tolist()
	return index_ADN_103
def scoreHomoAfterGM(results,ADN103): # score de test avec les ADN103 pour voir combient sont bien trouves sur les differentes plaques 
	clust1=0
	clust2=0
	clust3=0
	for i in ADN103:
		if results[i] == 0:
			clust1+=1
		if results[i] == 1:
			clust2+=1
		if results[i] == 2:
			clust3+=1
	score = max(clust1,clust2,clust3) / len(ADN103)
	return score
def effectifs(results): # get observed effectives of each genotype after GM by prediction 
	nbAA=0
	nbAB=0
	nbBB=0
	for i in results:
		if i==0:
			nbAA += 1
		if i==1:
			nbAB+=1
		if i==2:
			nbBB+=1
	effectifsAfterGM = [nbAA,nbAB,nbBB]
	return effectifsAfterGM
def getMAFobs(effectifs_predicted): # calcul MAF observed with effectifs predicted 
	freq_A = ((2 * effectifs_predicted[0]) + effectifs_predicted[1]) / (2 * (sum(effectifs_predicted)))
	freq_B = ((2 * effectifs_predicted[2]) + effectifs_predicted[1]) / (2 * (sum(effectifs_predicted)))
	if freq_A < freq_B:
		mafOBS = [freq_A,'A']
	else :
		mafOBS = [freq_B,'B']
	return mafOBS # return the smaller frequence = MAF
def getIndicePlateFromSNP(df): # get indices of plates by SNP
	listIndicesAllPlates = []
	df = df.reset_index() # reset index in a snp to have the same if GM later
	for name,group in df.groupby("Plate"): 
		sousListe = []
		sousListe.append(name)
		sousListe.append(group.index.tolist())
		listIndicesAllPlates.append(sousListe)
	return listIndicesAllPlates

#################### OUTPUTS ######################
def writeProbasConfidence(df,FileOut,probasGM): # write a file with probabilities and confidence score for each genotype 
	probaAA = pd.DataFrame(data=probasGM[0:,0:1],columns=["AA"])
	df["probaAA"] = probaAA['AA']
	probaAB = pd.DataFrame(data=probasGM[0:,1:2],columns=["AB"])
	df["probaAB"] = probaAB['AB']
	probaBB = pd.DataFrame(data=probasGM[0:,2:],columns=["BB"])
	df["probaBB"] = probaBB['BB']
	df["Confidences"] = 1-(df[["probaAA","probaAB","probaBB"]].max(axis=1))
	df.to_csv(path_or_buf="/home/e062531t/Bureau/Sidwell/res/probas_confidences.txt")
	return df
def writeCov(FileOut,covariances): # write a file with covariances matrix 
	fichier = open(FileOut, "w")
	fichier.write("AA"+"\t"+"\t"+"AB"+"\t"+"\t"+"BB"+"\t"+"\n")
	fichier.write(str(covariances[0][0][0])+"\t"+str(covariances[0][0][1])+"\t"+str(covariances[1][0][0])+"\t"+str(covariances[1][0][1])+"\t"+str(covariances[2][0][0])+"\t"+str(covariances[2][0][1])+"\n")
	fichier.write(str(covariances[0][1][0])+"\t"+str(covariances[0][1][1])+"\t"+str(covariances[1][1][0])+"\t"+str(covariances[1][1][1])+"\t"+str(covariances[2][1][0])+"\t"+str(covariances[2][1][1])+"\n")
def writeMeans(FileOut,means,genotypes): # write a file with means for each genotype 
	fichier = open(FileOut, "w")
	fichier.write("\t"+"LogRatio"+"\t"+"Strength"+"\n")
	for i, (mean, genotype) in enumerate(zip(means, genotypes)):
		fichier.write(genotype+"\t")
		for j in mean:
			fichier.write(str(j)+"\t")
		fichier.write("\n")
def plot_results_by_plate(datas,resultGM,means,covariances,name_snp,index_ADN_103,score,maf,HB,MAF_obs,listIndicesAllPlates): # plot results on graph 
    colors_genotypes = ['pink' if i==0 else 'skyblue' if i==1 else 'lightgreen' for i in resultGM]
    markers_genotypes = ['s' if i==0 else 'o' if i==1 else 'D' for i in resultGM]
    plt.clf()
    ax=plt.gca()
    for i, x, y, c, m in zip(range(len(datas)),datas[:,0], datas[:,1], colors_genotypes, markers_genotypes):
    	if i in listIndicesAllPlates[0][1]:
    		ax.scatter(datas[i,0], datas[i,1], alpha=0.8, c="aliceblue",marker=m, label="RAC1") # if indice is in the list of indices of plate 1 : set the color of the plate
    		if i in index_ADN_103:
    			ax.scatter(datas[i,0], datas[i,1], alpha=1, marker="*",c="aliceblue", s=100, label="RAC1") # and if it is also ADN103 : set a special marker with the color of the plate		
    	if i in listIndicesAllPlates[1][1]:
    		ax.scatter(datas[i,0], datas[i,1], alpha=0.8, c="red",marker=m, label="RAC10")
    		if i in index_ADN_103:
    			ax.scatter(datas[i,0], datas[i,1], alpha=1, marker="*",c="red", s=100, label="RAC10")
    	if i in listIndicesAllPlates[2][1]:
    		ax.scatter(datas[i,0], datas[i,1], alpha=0.8, c="blue",marker=m, label="RAC11-PVM1")
    		if i in index_ADN_103:
    			ax.scatter(datas[i,0], datas[i,1], alpha=1, marker="*",c="blue", s=100, label="RAC11-PVM1")
    	if i in listIndicesAllPlates[3][1]:
    		ax.scatter(datas[i,0], datas[i,1], alpha=0.8, c="green",marker=m, label="RAC12")
    		if i in index_ADN_103:
    			ax.scatter(datas[i,0], datas[i,1], alpha=1, marker="*",c="green", s=100, label="RAC12")
    	if i in listIndicesAllPlates[4][1]:
    		ax.scatter(datas[i,0], datas[i,1], alpha=0.8, c="yellow",marker=m, label="RAC13")
    		if i in index_ADN_103:
    			ax.scatter(datas[i,0], datas[i,1], alpha=1, marker="*",c="yellow", s=100, label="RAC13")
    	if i in listIndicesAllPlates[5][1]:
    		ax.scatter(datas[i,0], datas[i,1], alpha=0.8, c="purple",marker=m, label="RAC14")
    		if i in index_ADN_103:
    			ax.scatter(datas[i,0], datas[i,1], alpha=1, marker="*",c="purple", s=100, label="RAC14")
    	if i in listIndicesAllPlates[6][1]:
    		ax.scatter(datas[i,0], datas[i,1], alpha=0.8, c="orange",marker=m, label="RAC15")
    		if i in index_ADN_103:
    			ax.scatter(datas[i,0], datas[i,1], alpha=1, marker="*",c="orange", s=100, label="RAC15")
    	if i in listIndicesAllPlates[7][1]:
    		ax.scatter(datas[i,0], datas[i,1], alpha=0.8, c="white",marker=m, label="RAC2")
    		if i in index_ADN_103:
    			ax.scatter(datas[i,0], datas[i,1], alpha=1, marker="*",c="white", s=100, label="RAC2")
    	if i in listIndicesAllPlates[8][1]:
    		ax.scatter(datas[i,0], datas[i,1], alpha=0.8, c="black",marker=m, label="RAC3")
    		if i in index_ADN_103:
    			ax.scatter(datas[i,0], datas[i,1], alpha=1, marker="*",c="black", s=100, label="RAC3")
    	if i in listIndicesAllPlates[9][1]:
    		ax.scatter(datas[i,0], datas[i,1], alpha=0.8, c="antiquewhite",marker=m, label="RAC4")
    		if i in index_ADN_103:
    			ax.scatter(datas[i,0], datas[i,1], alpha=1, marker="*",c="antiquewhite", s=100, label="RAC4")
    	if i in listIndicesAllPlates[10][1]:
    		ax.scatter(datas[i,0], datas[i,1], alpha=0.8, c="beige",marker=m, label="RAC5")
    		if i in index_ADN_103:
    			ax.scatter(datas[i,0], datas[i,1], alpha=1, marker="*",c="beige", s=100, label="RAC5")
    	if i in listIndicesAllPlates[11][1]:
    		ax.scatter(datas[i,0], datas[i,1], alpha=0.8, c="bisque",marker=m, label="RAC6")
    		if i in index_ADN_103:
    			ax.scatter(datas[i,0], datas[i,1], alpha=1, marker="*",c="bisque", s=100, label="RAC6")
    	if i in listIndicesAllPlates[12][1]:
    		ax.scatter(datas[i,0], datas[i,1], alpha=0.8, c="blueviolet",marker=m, label="RAC7")
    		if i in index_ADN_103:
    			ax.scatter(datas[i,0], datas[i,1], alpha=1, marker="*",c="blueviolet", s=100, label="RAC7")
    	if i in listIndicesAllPlates[13][1]:
    		ax.scatter(datas[i,0], datas[i,1], alpha=0.8, c="brown",marker=m, label="RAC8")
    		if i in index_ADN_103:
    			ax.scatter(datas[i,0], datas[i,1], alpha=1, marker="*",c="brown", s=100, label="RAC8")
    	if i in listIndicesAllPlates[14][1]:
    		ax.scatter(datas[i,0], datas[i,1], alpha=0.8, c="chartreuse",marker=m, label="RAC9")
    		if i in index_ADN_103:
    			ax.scatter(datas[i,0], datas[i,1], alpha=1, marker="*",c="chartreuse", s=100, label="RAC9")
    	
    plt.title("Calling "+name_snp)
    plt.xlabel('LogRatio')
    plt.ylabel('Strength')
    nbgenos = effectifs(resultGM)
    #plt.legend(loc = 'upper right')
    plt.text(0.015, 0.98,"Score homogeneite = "+str(score),fontsize=8, ha='left', va='center', transform=ax.transAxes)
    plt.text(0.015, 0.95,"MAF_theo = "+str(maf[0])+"  allele:"+str(maf[1]),fontsize=8, ha='left', va='center', transform=ax.transAxes)
    plt.text(0.015, 0.92,"HB = "+str(HB),fontsize=8, ha='left', va='center', transform=ax.transAxes)
    plt.text(0.015, 0.89,"MAF_obs = "+str(MAF_obs[0])+"  allele:"+str(MAF_obs[1]),fontsize=8, ha='left', va='center', transform=ax.transAxes)
    plt.text(0.015, 0.86,"NbAA:"+str(nbgenos[0])+"  nbAB:"+str(nbgenos[1])+"  nbBB:"+str(nbgenos[2]),fontsize=8, ha='left', va='center', transform=ax.transAxes)
    #plt.show()
    plt.savefig(str(name_snp)+"_gm_corrected_priors_by_plate.png")
def plot_results(datas,resultGM,means,covariances,name_snp,index_ADN_103,score,maf,HB,MAF_obs): # plot results on graph 
    colors_genotypes = ['pink' if i==0 else 'skyblue' if i==1 else 'lightgreen' for i in resultGM]
    plt.clf()
    for i in range(len(means)):
    	ax=plt.gca()
    	ax.scatter(datas[:,0], datas[:,1], alpha=0.8, c=colors_genotypes)	
    	for j in index_ADN_103:
    		ax.scatter(datas[j,0], datas[j,1], alpha=1, c="black")

    plt.title("Calling "+name_snp)
    plt.xlabel('LogRatio')
    plt.ylabel('Strength')
    nbgenos = effectifs(resultGM)
    #plt.legend(loc = 'upper right')
    plt.text(0.015, 0.98,"Score homogeneite = "+str(score),fontsize=8, ha='left', va='center', transform=ax.transAxes)
    plt.text(0.015, 0.95,"MAF_theo = "+str(maf[0])+"  allele:"+str(maf[1]),fontsize=8, ha='left', va='center', transform=ax.transAxes)
    plt.text(0.015, 0.92,"HB = "+str(HB),fontsize=8, ha='left', va='center', transform=ax.transAxes)
    plt.text(0.015, 0.89,"MAF_obs = "+str(MAF_obs[0])+"  allele:"+str(MAF_obs[1]),fontsize=8, ha='left', va='center', transform=ax.transAxes)
    plt.text(0.015, 0.86,"NbAA:"+str(nbgenos[0])+"  nbAB:"+str(nbgenos[1])+"  nbBB:"+str(nbgenos[2]),fontsize=8, ha='left', va='center', transform=ax.transAxes)
    #plt.show()
    plt.savefig(str(name_snp)+"_gm_corrected.png")
def plot_results_by_plate_short(datas,resultGM,means,covariances,name_snp,index_ADN_103,score,maf,HB,MAF_obs,listIndicesAllPlates, df): # plot results on graph 
    colors_genotypes = ['pink' if i==0 else 'skyblue' if i==1 else 'lightgreen' for i in resultGM]
    markers_genotypes = ['s' if i==0 else 'o' if i==1 else 'D' for i in resultGM]
    print(markers_genotypes)
    colours = ["aliceblue", "red", "blue", "green", "yellow", "purple", "orange", "white", "black","antiquewhite","beige","bisque","blueviolet","brown","chartreuse"]
    plt.clf()
    ax=plt.gca()
    for (name, group), c in zip(df.groupby('Plate'), colours):
    	data = group[['CorrectedLogRatio','CorrectedStrength']]
    	#print(name)
    	indexPlate = data.index.tolist()
    	for i, m in zip(indexPlate, markers_genotypes):
    		ax.scatter(datas[i,0], datas[i,1], alpha=0.8, c=c, marker=m)
    	"""for x, y, m, in zip(data[:,0], data[:,1], markers_genotypes):
    		print(x,y,m)
    		ax.scatter(x, y, alpha=0.8, c=c, marker=m)"""
    	"""for (index,row), m in zip(data.iterrows(), markers_genotypes):
    		print (row[0],row[1], c, m)
    		ax.scatter(row[0], row[1], alpha=0.8, c=c, marker=m)"""
	  	
    plt.title("Calling "+name_snp)
    plt.xlabel('LogRatio')
    plt.ylabel('Strength')
    nbgenos = effectifs(resultGM)
    plt.text(0.015, 0.98,"Score homogeneite = "+str(score),fontsize=8, ha='left', va='center', transform=ax.transAxes)
    plt.text(0.015, 0.95,"MAF_theo = "+str(maf[0])+"  allele:"+str(maf[1]),fontsize=8, ha='left', va='center', transform=ax.transAxes)
    plt.text(0.015, 0.92,"HB = "+str(HB),fontsize=8, ha='left', va='center', transform=ax.transAxes)
    plt.text(0.015, 0.89,"MAF_obs = "+str(MAF_obs[0])+"  allele:"+str(MAF_obs[1]),fontsize=8, ha='left', va='center', transform=ax.transAxes)
    plt.text(0.015, 0.86,"NbAA:"+str(nbgenos[0])+"  nbAB:"+str(nbgenos[1])+"  nbBB:"+str(nbgenos[2]),fontsize=8, ha='left', va='center', transform=ax.transAxes)
    #plt.show()
    plt.savefig(str(name_snp)+"_gm_priors_corrected_by_plate.png")
def plot_results_with_ellipses(datas,resultGM,means,covariances,name_snp,index_ADN_103,score,maf,HB,MAF_obs): # plot results on graph 
    colors_genotypes = ['pink' if i==0 else 'skyblue' if i==1 else 'lightgreen' for i in resultGM]
    markers_genotypes = ['x','o','D']
    colors_ellipses = ['navy', 'cyan', 'cornflowerblue']
    plt.clf()
    for i in range(len(means)):
    	v, w = linalg.eigh(covariances[i])
    	v = 2. * np.sqrt(2.) * np.sqrt(v)
    	u = w[0] / linalg.norm(w[0])
    	ax=plt.gca()
    	ax.scatter(datas[:,0], datas[:,1], alpha=0.8, c=colors_genotypes,marker='x')
    	plt.show()  	
    	for i in index_ADN_103:
    		ax.scatter(datas[i,0], datas[i,1], alpha=1, c="black")

    	# Plot an ellipse to show the Gaussian component
    	angle = np.arctan(u[1] / u[0])
    	angle = 180. * angle / np.pi  # convert to degrees
    	ell = mpl.patches.Ellipse(means[i], v[0], v[1], 180. + angle, color=colors_ellipses[i])
    	ell.set_clip_box(ax.bbox)
    	ell.set_alpha(0.5)
    	ax.add_artist(ell)

    plt.title("Calling "+name_snp)
    plt.xlabel('LogRatio')
    plt.ylabel('Strength')
    plt.text(0.015, 0.98,"Score homogeneite = "+str(score),fontsize=8, ha='left', va='center', transform=ax.transAxes)
    plt.text(0.015, 0.95,"MAF_theo = "+str(maf[0])+"  allele:"+str(maf[1]),fontsize=8, ha='left', va='center', transform=ax.transAxes)
    plt.text(0.015, 0.92,"HB = "+str(HB),fontsize=8, ha='left', va='center', transform=ax.transAxes)
    plt.text(0.015, 0.89,"MAF_obs = "+str(MAF_obs[0])+"  allele:"+str(MAF_obs[1]),fontsize=8, ha='left', va='center', transform=ax.transAxes)
    plt.show()
    #plt.savefig(str(name_snp)+"_gm_priors_corrected.png")

############### GAUSSIAN MIXTURE #############
def GaussianMixture(df,wgts,means,genotypes,name_snp,index_ADN_103,maf,priors,pvar,listIndicesAllPlates): # do the GM with EM algorithm and output results 
	if args['--correction'] == 'True':
		LogRatio = "CorrectedLogRatio"
		Strength = "CorrectedStrength"
		means_initialisation = priors
	else:
		LogRatio = "Log Ratio"
		Strength = "Strength"
		means_initialisation = means

	datas = df[[LogRatio, Strength]].as_matrix()
	precisions = np.diag([pvar]*2)
	mafHB = float(maf[0])
	freqBB = mafHB**2
	freqAB = 2 * mafHB * (1 - mafHB)
	freqAA = (1-mafHB)**2
	weigths_theo = np.asarray([freqAA,freqAB,freqBB])
	gm = mixture.GaussianMixture(n_components=3, covariance_type="tied",weights_init=weigths_theo,means_init=means_initialisation,random_state=0, max_iter=200, verbose=0, warm_start=False).fit(datas)
	
	resultatsGM = gm.predict(datas)
	probasGM = gm.predict_proba(datas)
	cov_res = gm.covariances_
	if len(cov_res) < 3:
		#if cov_type = tied cause components share the same matrix but need a square matrix
		cov_res = np.array([cov_res]*3)
	means_res = gm.means_

	#AIC = gm.aic(datas)
	#BIC = gm.bic(datas)

	##### OUTPUT RESULTS #####
	score = scoreHomoAfterGM(resultatsGM,index_ADN_103)
	effectifs_after_GM = effectifs(resultatsGM)
	MAF_obs = getMAFobs(effectifs_after_GM)
	HB = hardyWeinberg(len(df),effectifs_after_GM,maf,MAF_obs)
	plot_results(datas,resultatsGM,means_res, cov_res,name_snp,index_ADN_103,score,maf,HB,MAF_obs)
	#plot_results_by_plate(datas,resultatsGM,means_res,cov_res,name_snp,index_ADN_103,score,maf,HB,MAF_obs,listIndicesAllPlates,df)
	#probas = writeProbasConfidence(df,"/home/e062531t/Bureau/Sidwell/res/probas_confidence_genotypes.txt", probasGM)
	#writeCov("/home/e062531t/Bureau/Sidwell/res/cov_gm.txt",cov_res) 
	#writeMeans("/home/e062531t/Bureau/Sidwell/res/means_gm.txt",means_res, genotypes)

################# DEBUGAGE ##################
def plotByPlate(datas,wgts,means,genotypes,name_snp): # plot results GM by plate 
	#colours = ["aliceblue", "red", "blue", "green", "yellow", "purple", "orange", "white", "black","antiquewhite","beige","bisque","blueviolet","brown","chartreuse"]
	#colours2 = ["white", "red", "white", "white", "white", "white", "white", "blue", "yellow","white","white","white","blueviolet","white","white"]
	for name,group in datas.groupby("Plate",sort=False):
	#for i,((name,group),color) in enumerate(zip(datas.groupby("Plate",sort=False),colours2)):
		df = group[["Log Ratio","Strength"]]
		#df = group[["CorrectedLogRatio","CorrectedStrength"]]
		df=df.as_matrix()
		gm = mixture.GaussianMixture(n_components=3, covariance_type="tied", weights_init=wgts,means_init=means,random_state=0, max_iter=200, verbose=0, warm_start=False).fit(df)
		colors_genotypes = ['r' if i==0 else 'b' if i==1 else 'g' for i in gm.predict(df)]
		ax=plt.gca()
		#clear permet d'avoir un graphe different par plaque avec une plaque par graphe sinon les plaques precedentes restent sur le graphe
		#ax.clear()
		ax.scatter(df[:,0], df[:,1], alpha=0.8, c=colors_genotypes)		
		#ax.set_xlim([-4,2])
		#ax.set_ylim([-1.5,1])
	#plt.savefig(str(name)+str(name_snp)+"_gm_corrected_by_plate.png")
	plt.savefig(str(name_snp)+"by_plate_"+"gm_corrected.png")
	#plt.show()

################## HARDY WEINBERG #################
def hardyWeinberg(N,effectifs,maf,maf_obs): # return a chi2 
	maf = float(maf[0])
	maf_obs = maf_obs[0]
	theo_eff_homo_variant = N * (maf_obs**2)
	theo_eff_AB = 2 * N * maf_obs * (1 - maf_obs)
	theo_eff_homo_ref = N * ((1-maf_obs)**2)
	theo_effectifs = [theo_eff_homo_ref, theo_eff_AB, theo_eff_homo_variant]
	chi2 = (((theo_eff_homo_ref-effectifs[0])**2)/effectifs[0])+(((theo_eff_AB-effectifs[1])**2)/effectifs[1])+(((theo_eff_homo_variant-effectifs[2])**2)/effectifs[2])
	pval = (scipy.stats.chisquare(effectifs, f_exp=theo_effectifs, ddof=1)[1])
	return pval

################ SIMULATIONS ################
def generateMultivarie(maf,n,size,df): # generate multivarie for simulations 
	freq_allelique=maf[0]
	df=df.rename(columns = {'Forced Call':'Forced_Call'})
	mu1 = (df[df.Forced_Call=="AA"].mean()["Strength"])
	mu2 = (df[df.Forced_Call=="AB"].mean()["Strength"])
	mu3 = (df[df.Forced_Call=="BB"].mean()["Strength"])
	v1_strength = (df[df.Forced_Call=="AA"].var()["Strength"])
	v2_strength = (df[df.Forced_Call=="AB"].var()["Strength"])
	v3_strength = (df[df.Forced_Call=="BB"].var()["Strength"])
	#print(v1_strength,v2_strength,v3_strength)
	for name, group in df.groupby("Plate"):
		pvals = [(1-freq_allelique)**2, 2*freq_allelique*(1-freq_allelique), freq_allelique**2]
		tirage = np.random.multinomial(n,pvals,size)
		n1 = tirage[0][0]
		n2 = tirage[0][1]
		n3 = tirage[0][2]
		m1 = (df[df.Forced_Call=="AA"].mean()["Log Ratio"])
		m2 = (df[df.Forced_Call=="AB"].mean()["Log Ratio"])
		m3 = (df[df.Forced_Call=="BB"].mean()["Log Ratio"])
		v1 = (df[df.Forced_Call=="AA"].var()["Log Ratio"])
		v2 = (df[df.Forced_Call=="AB"].var()["Log Ratio"])
		v3 = (df[df.Forced_Call=="BB"].var()["Log Ratio"])
		x = [m1,m2,m3]
		z = [v1,v2,v3]
		y = [mu1,mu2,mu1]
		X = np.vstack((x,y))
		sigma = np.cov(X)
		#print("mean_strength:",mu1,mu2,mu3," mean_log:",x," var_log:",z," cov_log:",sigma)
		multivarie_1 = np.random.multivariate_normal([m1,mu1],sigma,n1)
		multivarie_2 = np.random.multivariate_normal([m2,mu2],sigma,n2)
		multivarie_3 = np.random.multivariate_normal([m3,mu1],sigma,n3)
		print("name :",name," AA:",multivarie_1)
		print("AB:",multivarie_2)
		print("BB:",multivarie_3)

############## REDEFINE PRIORS ##############
def redefinePriors(df_snp,maf): # define priors from maf to re genotype after 
	N = len(df_snp)
	maf_num = maf[0]

	nbAB = 2 * N * maf_num * (1 - maf_num)
	nbAB = int(nbAB)

	if maf[1] == 'A':
		nbAA = N * (maf_num**2)
		nbAA = int(nbAA)
		nbBB = N * ((1-maf_num)**2)
		nbBB = int(nbBB)
		df_snp = df_snp.sort_values(['CorrectedLogRatio'],ascending=True) # small values on top
		new_priorBB_log = df_snp.iloc[int((0+nbBB)/2)]['CorrectedLogRatio']
		priorBB_strength = df_snp.iloc[int((0+nbBB)/2)]['CorrectedStrength']
		b = int(nbAA/2)
		new_priorAA_log = df_snp.iloc[nbBB+nbAB+b]['CorrectedLogRatio']
		priorAA_strength = df_snp.iloc[nbBB+nbAB+b]['CorrectedStrength']

	if maf[1] == 'B':
		df_snp = df_snp.sort_values(['CorrectedLogRatio'],ascending=False) # high values on top
		nbBB = N * (maf_num**2)
		nbBB = int(nbBB)
		nbAA = N * ((1-maf_num)**2)
		nbAA = int(nbAA)
		new_priorAA_log = df_snp.iloc[int((0+nbAA)/2)]['CorrectedLogRatio'] # parmi le nombre theorique de AA : prend le point avec un logRatio median
		priorAA_strength = df_snp.iloc[int((0+nbAA)/2)]['CorrectedStrength']
		b = int(nbBB/2)
		new_priorBB_log = df_snp.iloc[nbAA+nbAB+b]['CorrectedLogRatio'] # parmi le nombre theorique de BB (on ajoute le nb de AA et de BB pour arriver a lindice)
		priorBB_strength = df_snp.iloc[nbAA+nbAB+b]['CorrectedStrength']

	a = int(nbAB/2)
	new_priorAB_log = df_snp.iloc[nbAA+a]['CorrectedLogRatio']
	priorAB_strength = df_snp.iloc[nbAA+a]['CorrectedStrength']

	priors = [[new_priorAA_log,priorAA_strength],[new_priorAB_log,priorAB_strength],[new_priorBB_log,priorBB_strength]]
	priors = np.asarray(priors)

	return priors


################## MAIN ################
def main():
	
	genotypes = ["AA","AB","BB"]
	AllDatas = importDatas(args['--callingRAC'])
	AllDatas = AllDatas.sort_values(['snp','Sample'], ascending=[True,False]) #sort by snp to keep indices to plot after
	
	listSNP = import_list_snp(args['--snpList'])

	#listSNP = ['rs7074353']
	#listSNP = ['rs35386136']

	for snp in listSNP:
		df_snp = degPlatesFromSNP(AllDatas,snp)
		listIndicesAllPlates = getIndicePlateFromSNP(df_snp)
		try:
			maf = getMAF("all_maf_4_GM.txt",snp)
		except FileNotFoundError:
			print("Run getMAF.py first to get MAF from annotation file")
			break	
		if args['--correction'] == 'True':
			df_to_GM = correct_logRatio_centrality_etendue(correct_strength_median(df_snp))
			new_priors = redefinePriors(df_to_GM,maf)
		else:
			df_to_GM = df_snp
			new_priors = 0	

		 #generateMultivarie(maf,96,1,df_snp)

		wgtsGenotypes,meansGenotypes,countGenotypes,pvar = getParamsFromDatas(df_to_GM,genotypes)

		print(snp)
		
		df_to_GM = df_to_GM.reset_index()
		listADN = getADN103(df_to_GM)

		GaussianMixture(df_to_GM,wgtsGenotypes,meansGenotypes,genotypes,snp,listADN,maf,new_priors,pvar,listIndicesAllPlates)

	
	
###### correction : modify GaussianMixture, GetParamsFromDatas, redefinePriors, main ###########

if __name__ == '__main__' :
	args = docopt.docopt(__doc__, version=__version__)
	main()

	
	
	



	



