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
    -h --help     Show help.
    snpList		list snp txt
    callingRAC		results snp

"""

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
pd.set_option('chained',None) #pour eviter le warning sur les copies de dataframe


################ IMPORT DATAS #################
def importDatas(fileToImport): # import file in dataframe and add header, chose wich one 
	# import datas as dataframe with pandas
	df = pd.read_csv(fileToImport, sep='\t')
	#CHOSE HEADER
	#df.columns = ['Plate','Sample','ProbeSetName','Call','Confidence','Log Ratio','Strength','Forced Call'] #Mettre en commentaire si pas de header
	df.columns = ['Plate','snp','Sample','ProbeSetName','Call','Confidence','Log Ratio','Strength','Forced Call']
	return df
def degPlatesFromSNP(df,name_snp): # split dataframe by snp 
	table_snp = df.loc[df.snp==name_snp]
	return table_snp
def import_list_snp(listSNP):
    list_snp = []
    with open(listSNP,'r') as f:
        for snp in f.readlines():
            snp = snp.replace("\n","")
            list_snp.append(snp)
    return list_snp
def getMAF(fileAllMAF,name_snp):
	dfMAF = pd.read_csv(fileAllMAF)
	MAFsnp = dfMAF.loc[dfMAF.snp==name_snp]['maf'].item()
	return float(MAFsnp)

############## HOMOGENEIZATION PLATES (CORRECTION) #############
def quartile_subset(logratios,lower,upper): # avoid outliers 
    #subset logRatios to not count extremes in mean
    #logratio v must be higher than the lower quantile and lower than the upper quantile to avoid outliers
    return logratios.loc[[True if v < logratios.quantile(q=upper) and v > logratios.quantile(q=lower) else False for v in logratios]]
def dist_inter_quantile(serie,lower,upper): # distance between two quantiles 
	dist = serie.quantile(q=upper) - serie.quantile(q=lower)
	return dist
def correct_logRatio_median_interquantile(df): # correct log ratio by median or mean and by quantile -> Do not give good results 
	#homogeneisation des plaques : correction du log ratio par la moyenne par plaque et par les quantiles, correction de Strength par la moyenne
	df['medianLogRatioPlate'] = df.groupby('Plate')['Log Ratio'].transform('median')
	df['meanLogRatioPlate'] = df.groupby('Plate')['Log Ratio'].transform('mean')
	df['q75Plate'] = df.groupby('Plate')['Log Ratio'].transform(lambda x: x.quantile(q=0.75))
	df['q25Plate'] = df.groupby('Plate')['Log Ratio'].transform(lambda x: x.quantile(q=0.25))
	df['CorrectedLogRatioMedian'] = df['Log Ratio'] - df.groupby('Plate')['Log Ratio'].transform('mean') #cest cense etre une correction par la mediane mais ca marche mieux avec la moyenne
	df['CorrectedLogRatio'] = df['CorrectedLogRatioMedian'] * 1.34 / df.groupby('Plate')['Log Ratio'].transform(lambda x: (x.quantile(q=0.75) - x.quantile(q=0.25)))
	return df
def correct_logRatio_centrality_etendue(df): # correct log ratio by centrality and extended -> good correction with cool snp 
	df['m1'] = df.groupby('Plate')['Log Ratio'].transform(lambda s: quartile_subset(s,0.20,0.80).mean())
	df['m2'] = df.groupby('Plate')['Log Ratio'].transform(lambda s: quartile_subset(s,0.25,0.75).mean())
	df['m3'] = df.groupby('Plate')['Log Ratio'].transform(lambda s: quartile_subset(s,0.30,0.70).mean())
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
		#### !!!!!!!!!!!  choose "Log Ratio" or "CorrectedLogRatio"
		#meanLog = (df[df.Forced_Call==i].mean()["Log Ratio"])
		meanLog = (df[df.Forced_Call==i].mean()["CorrectedLogRatio"]) #mean of logRatio for genotype i
		meansCouple.append(meanLog)
		#### !!!!!!!!!!!  choose "Strength" or "CorrectedStrength"
		#meanStrength = (df[df.Forced_Call==i].mean()["Strength"])
		meanStrength = (df[df.Forced_Call==i].mean()["CorrectedStrength"]) #mean of Strength for genotype i
		meansCouple.append(meanStrength)
		means.append(meansCouple)
		#pvar=np.var(df["Strength"])
		pvar=np.var(df["CorrectedStrength"])
	wgts = np.asarray(wgts)
	means_array = np.asarray(means)
	return wgts, means_array, countGeno, pvar
def getADN103(df): # recupere les indices des ADN103 par snp pour les colorer ensuite sur le graph 
	expreg = "ADN103_RAC."
	index_ADN_103 = df[df['Sample'].str.contains(expreg, regex=True)].index.tolist()
	return index_ADN_103
def scoreHomogeneite(df): # score de test avec les ADN103 pour voir combient sont bien trouves sur les differentes plaques 
	nbAA=0
	nbAB=0
	nbBB=0
	allADN103 = getADN103(df)
	for i in allADN103:
		if df['Forced Call'][i] == "AA":
			nbAA+=1
		if df['Forced Call'][i] == "AB":
			nbAB+=1
		if df['Forced Call'][i] == "BB":
			nbBB+=1
	score = max(nbAA,nbAB,nbBB) / len(allADN103)
	return score
def scoreHomoAfterGM(results,ADN103):
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
def effectifs(results):
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
def plot_results(datas,resultGM,means,covariances,name_snp,index_ADN_103,score,maf,HB): # plot results on graph 
    colors_genotypes = ['pink' if i==0 else 'skyblue' if i==1 else 'lightgreen' for i in resultGM]
    colors_ellipses = ['navy', 'cyan', 'cornflowerblue']
    plt.clf()
    for i in range(len(means)):
    	v, w = linalg.eigh(covariances[i])
    	v = 2. * np.sqrt(2.) * np.sqrt(v)
    	u = w[0] / linalg.norm(w[0])
    	ax=plt.gca()
    	ax.scatter(datas[:,0], datas[:,1], alpha=0.8, c=colors_genotypes)  	
    	for i in index_ADN_103:
    		ax.scatter(datas[i,0], datas[i,1], alpha=1, c="black")

    	"""# Plot an ellipse to show the Gaussian component
    	angle = np.arctan(u[1] / u[0])
    	angle = 180. * angle / np.pi  # convert to degrees
    	ell = mpl.patches.Ellipse(means[i], v[0], v[1], 180. + angle, color=colors_ellipses[i])
    	ell.set_clip_box(ax.bbox)
    	ell.set_alpha(0.5)
    	ax.add_artist(ell)"""

    plt.title("Calling "+name_snp)
    plt.xlabel('LogRatio')
    plt.ylabel('Strength')
    plt.text(0.015, 0.95,"Score homogeneite = "+str(score), ha='left', va='center', transform=ax.transAxes)
    plt.text(0.015, 0.90,"MAF = "+str(maf), ha='left', va='center', transform=ax.transAxes)
    plt.text(0.015, 0.85,"HB = "+str(HB), ha='left', va='center', transform=ax.transAxes)
    #plt.show()
    plt.savefig(str(name_snp)+"_gm_corrected.png")

############### GAUSSIAN MIXTURE #############
def GaussianMixture(df,wgts,means,genotypes,name_snp,index_ADN_103,maf,priors,pvar): # do the GM with EM algorithm and output results 
	#datas = df[['Log Ratio','Strength']].as_matrix()
	datas = df[['CorrectedLogRatio','CorrectedStrength']].as_matrix()
	precisions = np.diag([pvar]*2)
	maf = float(maf)
	freqBB = maf**2
	freqAB = 2 * maf * (1 - maf)
	freqAA = (1-maf)**2
	weigths_theo = np.asarray([freqAA,freqAB,freqBB])
	gm = mixture.GaussianMixture(n_components=3, covariance_type="tied",weights_init=weigths_theo,means_init=means,precisions_init=precisions,random_state=0, max_iter=200, verbose=0, warm_start=False).fit(datas)
	
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
	HB = hardyWeinberg(len(df),effectifs_after_GM,maf)
	plot_results(datas,resultatsGM,means_res, cov_res,name_snp,index_ADN_103,score,maf,HB)
	#probas = writeProbasConfidence(df,"/home/e062531t/Bureau/Sidwell/res/probas_confidence_genotypes.txt", probasGM)
	#writeCov("/home/e062531t/Bureau/Sidwell/res/cov_gm.txt",cov_res) 
	#writeMeans("/home/e062531t/Bureau/Sidwell/res/means_gm.txt",means_res, genotypes)
def HMM(df,wgts,means,genotypes,name_snp,index_ADN_103,maf,priors): # do the GM with EM algorithm and output results 
	#datas = df[['Log Ratio','Strength']].as_matrix()
	datas = df[['CorrectedLogRatio','CorrectedStrength']].as_matrix()
	
	model = hmm.GaussianHMM(n_components=3, covariance_type="tied",random_state=0)
	
	model.startprob_ = priors
	model.means_ = means

	resultatsGM = model.predict(datas)
	probasGM = model.predict_proba(datas)
	cov_res = model.covariances_
	if len(cov_res) < 3:
		#if cov_type = tied cause components share the same matrix but need a square matrix
		cov_res = np.array([cov_res]*3)
	means_res = model.means_
def BayesianGaussianMixture(df,wgts,means,genotypes,name_snp,index_ADN_103,maf,priors): # do the GM with EM algorithm and output results 
	#datas = df[['Log Ratio','Strength']].as_matrix()
	datas = df[['CorrectedLogRatio','CorrectedStrength']].as_matrix()
	gm = mixture.BayesianGaussianMixture(n_components=3, covariance_type="tied",mean_prior=priors,random_state=0, max_iter=200, verbose=0, warm_start=False).fit(datas)
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
	score = scoreHomogeneite(df)
	plot_results(datas,resultatsGM,means_res, cov_res,name_snp,index_ADN_103,score,maf)
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
def fequenceAlleles(df,wgts): # calcul frequence allelic from an all dataframe with all paltes 
	freq_AA = wgts[0]
	freq_AB = wgts[1]
	freq_BB = wgts[2]
	freq_A = freq_AA + ((1/2) * freq_AB)
	freq_B = freq_BB + ((1/2) * freq_AB)
	return freq_A, freq_B
def hardyWeinberg(N,effectifs,maf): # return a chi2 
	maf = float(maf)
	theo_eff_BB = N * (maf**2)
	theo_eff_AB = 2 * N * maf * (1 - maf)
	theo_eff_AA = N * ((1-maf)**2)
	theo_effectifs = [theo_eff_AA, theo_eff_AB, theo_eff_BB]
	pval = (scipy.stats.chisquare(effectifs, f_exp=theo_effectifs)[1])
	return pval
def desequilibreLiaison(df,wgts): # INCOMPLETE -> return DL but not sure of what it is 
	freq_A, freq_B = fequenceAlleles(df,wgts)
	DL = 0

################ SIMULATIONS ################
def generateMultivarie(freq_allelique,n,size,df): # generate multivarie for simulations 
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
		"""print("name :",name," AA:",multivarie_1)
		print("AB:",multivarie_2)
		print("BB:",multivarie_3)"""

############## REDEFINE PRIORS ##############
def redefinePriors(df_snp,maf): # define priors from maf to re genotype after 
	N = len(df_snp)

	nbBB = N * (maf**2)
	nbAB = 2 * N * maf * (1 - maf)
	nbAA = N * ((1-maf)**2)
	nbAA = int(nbAA)
	nbAB = int(nbAB)
	nbBB = int(nbBB)

	df_snp = df_snp.sort_values(['CorrectedLogRatio'])

	new_priorBB_log = df_snp.iloc[int((0+nbBB)/2)]['CorrectedLogRatio']
	priorBB_strength = df_snp.iloc[int((0+nbBB)/2)]['CorrectedStrength']

	a = int(nbAB/2)
	new_priorAB_log = df_snp.iloc[nbBB+a]['CorrectedLogRatio']
	priorAB_strength = df_snp.iloc[nbBB+a]['CorrectedStrength']

	b = int(nbAA/2)
	new_priorAA_log = df_snp.iloc[nbBB+nbAB+b]['CorrectedLogRatio']
	priorAA_strength = df_snp.iloc[nbBB+nbAB+b]['CorrectedStrength']

	priors = [[new_priorAA_log,priorAA_strength],[new_priorAB_log,priorAB_strength],[new_priorBB_log,priorBB_strength]]
	#priors = [new_priorAA_log,new_priorAB_log,new_priorBB_log]
	#print(priors)
	priors = np.asarray(priors)

	return priors


################## MAIN ################
def main():
	args = docopt.docopt(__doc__)
	genotypes = ["AA","AB","BB"]
	AllDatas = importDatas(args['--callingRAC'])
	AllDatas = AllDatas.sort_values(['snp','Sample'], ascending=[True,False]) #sort by snp to keep indices to plot after
	
	listSNP = import_list_snp(args['--snpList'])

	#listSNP = ['rs7074353']

	for snp in listSNP:
		
		df_snp = degPlatesFromSNP(AllDatas,snp)
		df_corrected = correct_logRatio_centrality_etendue(correct_strength_median(df_snp))
		wgtsGenotypes,meansGenotypes,countGenotypes,pvar = getParamsFromDatas(df_corrected,genotypes)

		try:
			maf = getMAF("all_maf_4_GM.txt",snp)
		except FileNotFoundError:
			print("Run getMAF.py first to get MAF from annotation file")
			break	

		print(snp)
		new_priors = redefinePriors(df_corrected,maf)
		
		df_corrected = df_corrected.reset_index()
		listADN = getADN103(df_corrected)
		
		GaussianMixture(df_corrected,wgtsGenotypes,meansGenotypes,genotypes,snp,listADN,maf,new_priors,pvar)

		#plotByPlate(df_snp,wgtsGenotypes,meansGenotypes,genotypes,snp)
	
	#generateMultivarie(0.27,96,1,AllDatas)


###### correction : modify GaussianMixture, GetParamsFromDatas, redefinePriors, main ###########

if __name__ == '__main__' :
	main()

	
	
	



	



