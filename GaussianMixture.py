#!/usr/bin/python3

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import mixture # Sklearn v0.18
import time
import itertools
from scipy import linalg
import scipy
import matplotlib as mpl
pd.set_option('chained',None)


################ IMPORT DATAS #################
def importDatas(fileToImport) :
	# import datas as dataframe with pandas
	df = pd.read_csv(fileToImport, sep='\t')
	#CHOSE HEADER
	#df.columns = ['Plate','Sample','ProbeSetName','Call','Confidence','Log Ratio','Strength','Forced Call'] #Mettre en commentaire si pas de header
	df.columns = ['Plate','snp','Sample','ProbeSetName','Call','Confidence','Log Ratio','Strength','Forced Call']
	return df
def degPlatesFromSNP(df,name_snp):
	table_snp = df.loc[df.snp==name_snp]
	return table_snp

##### HOMOGENEIZATION PLATES (CORRECTION) #####
def quartile_subset(logratios,lower,upper):
    #subset logRatios to not count extremes in mean
    #logratio v must be higher than the lower quantile and lower than the upper quantile to avoid outliers
    return logratios.loc[[True if v < logratios.quantile(q=upper) and v > logratios.quantile(q=lower) else False for v in logratios]]
def dist_inter_quantile(serie,lower,upper):
	dist = serie.quantile(q=upper) - serie.quantile(q=lower)
	return dist
def correct_logRatio_median_interquantile(df): # Do not give good results 
	#homogeneisation des plaques : correction du log ratio par la moyenne par plaque et par les quantiles, correction de Strength par la moyenne
	df['medianLogRatioPlate'] = df.groupby('Plate')['Log Ratio'].transform('median')
	df['meanLogRatioPlate'] = df.groupby('Plate')['Log Ratio'].transform('mean')
	df['q75Plate'] = df.groupby('Plate')['Log Ratio'].transform(lambda x: x.quantile(q=0.75))
	df['q25Plate'] = df.groupby('Plate')['Log Ratio'].transform(lambda x: x.quantile(q=0.25))
	df['CorrectedLogRatioMedian'] = df['Log Ratio'] - df.groupby('Plate')['Log Ratio'].transform('mean') #cest cense etre une correction par la mediane mais ca marche mieux avec la moyenne
	df['CorrectedLogRatio'] = df['CorrectedLogRatioMedian'] * 1.34 / df.groupby('Plate')['Log Ratio'].transform(lambda x: (x.quantile(q=0.75) - x.quantile(q=0.25)))
	return df
def correct_logRatio_centrality_etendue(df):
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
def correct_strength_median(df):
	df['medianStrengthPlate'] = df.groupby('Plate')['Strength'].transform('median')
	df['CorrectedStrength'] = df['Strength'] - df.groupby('Plate')['Strength'].transform('median')
	return df

###### PARAMETERS TO DO GAUSSIAN MIXTURE ######
def getParamsFromDatas(df_corrected, genotypes) : # Get weigths for each genotype and means for each genotype for LogRatio and Strength -> choose datas corrected per plate or not 
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
	wgts = np.asarray(wgts)
	means = np.asarray(means)
	return wgts, means, countGeno
def getADN103(df):
	expreg = "ADN103_RAC."
	index_ADN_103 = df[df['Sample'].str.contains(expreg, regex=True)].index.tolist()
	return index_ADN_103
def scoreHomogeneite(df):
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
def getMAF_for_snp(filepath,name_snp):
	df = pd.read_csv(filepath,sep=',',header=19,low_memory=False)
	line = df[df['dbSNP RS ID']==name_snp]['Minor Allele Frequency']
	line = line.str.split("/")
	allele_frequency=line.iloc[0][0]
	return allele_frequency

############## OUTPUTS ###############
def writeProbasConfidence(df,FileOut, probasGM) :
	probaAA = pd.DataFrame(data=probasGM[0:,0:1],columns=["AA"])
	df["probaAA"] = probaAA['AA']
	probaAB = pd.DataFrame(data=probasGM[0:,1:2],columns=["AB"])
	df["probaAB"] = probaAB['AB']
	probaBB = pd.DataFrame(data=probasGM[0:,2:],columns=["BB"])
	df["probaBB"] = probaBB['BB']
	df["Confidences"] = 1-(df[["probaAA","probaAB","probaBB"]].max(axis=1))
	df.to_csv(path_or_buf="/home/e062531t/Bureau/Sidwell/res/probas_confidences.txt")
def writeCov(FileOut, covariances) :
	fichier = open(FileOut, "w")
	fichier.write("AA"+"\t"+"\t"+"AB"+"\t"+"\t"+"BB"+"\t"+"\n")
	fichier.write(str(covariances[0][0][0])+"\t"+str(covariances[0][0][1])+"\t"+str(covariances[1][0][0])+"\t"+str(covariances[1][0][1])+"\t"+str(covariances[2][0][0])+"\t"+str(covariances[2][0][1])+"\n")
	fichier.write(str(covariances[0][1][0])+"\t"+str(covariances[0][1][1])+"\t"+str(covariances[1][1][0])+"\t"+str(covariances[1][1][1])+"\t"+str(covariances[2][1][0])+"\t"+str(covariances[2][1][1])+"\n")
def writeMeans(FileOut,means, genotypes) :
	fichier = open(FileOut, "w")
	fichier.write("\t"+"LogRatio"+"\t"+"Strength"+"\n")
	for i, (mean, genotype) in enumerate(zip(means, genotypes)):
		fichier.write(genotype+"\t")
		for j in mean:
			fichier.write(str(j)+"\t")
		fichier.write("\n")
def plot_results(datas, resultGM, means, covariances,name_snp):
    colors_genotypes = ['r' if i==0 else 'b' if i==1 else 'g' for i in resultGM]
    colors_ellipses = ['navy', 'cyan', 'cornflowerblue']
    for i, (mean, covar, color) in enumerate(zip(means, covariances, colors_ellipses)):
    	v, w = linalg.eigh(covar)
    	v = 2. * np.sqrt(2.) * np.sqrt(v)
    	u = w[0] / linalg.norm(w[0])
    	ax=plt.gca()
    	ax.scatter(datas[:,0], datas[:,1], alpha=0.8, c=colors_genotypes)

    	"""# Plot an ellipse to show the Gaussian component
    	angle = np.arctan(u[1] / u[0])
    	angle = 180. * angle / np.pi  # convert to degrees
    	ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
    	ell.set_clip_box(ax.bbox)
    	ell.set_alpha(0.5)
    	ax.add_artist(ell)"""

    #plt.show()
    plt.savefig(str(name_snp)+"_gm_corrected.png")
def plot_results2(datas, resultGM, means, covariances,name_snp,index_ADN_103,score,maf):
    colors_genotypes = ['r' if i==0 else 'b' if i==1 else 'g' for i in resultGM]
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
    	ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
    	ell.set_clip_box(ax.bbox)
    	ell.set_alpha(0.5)
    	ax.add_artist(ell)"""

    plt.title("Calling")
    plt.xlabel('LogRatio')
    plt.ylabel('Strength')
    plt.text(-11,1.3,"Score homogeneite = "+str(score))
    plt.text(-11,1.2,"MAF = "+str(maf))
    plt.show()
    #plt.savefig(str(name_snp)+"_gm_corrected.png")

############### GAUSSIAN MIXTURE #############
def GaussianMixture(df,wgts,means,genotypes,name_snp,index_ADN_103,maf) :
	#datas = df[['Log Ratio','Strength']].as_matrix()
	datas = df[['CorrectedLogRatio','CorrectedStrength']].as_matrix()
	gm = mixture.GaussianMixture(n_components=3, covariance_type="tied", weights_init=wgts,means_init=means,random_state=0, max_iter=200, verbose=0, warm_start=False).fit(datas)
	
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
	plot_results2(datas,resultatsGM,means_res, cov_res,name_snp,index_ADN_103,score,maf)
	#writeProbasConfidence(df,"/home/e062531t/Bureau/Sidwell/res/probas_confidence_genotypes.txt", probasGM)
	#writeCov("/home/e062531t/Bureau/Sidwell/res/cov_gm.txt",cov_res) 
	#writeMeans("/home/e062531t/Bureau/Sidwell/res/means_gm.txt",means_res, genotypes)

################# DEBUGAGE ##################
def plotByPlate(datas,wgts,means,genotypes,name_snp):
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

######### HARDY WEINBERG #########
def fequenceAlleles(df,wgts):
	freq_AA = wgts[0]
	freq_AB = wgts[1]
	freq_BB = wgts[2]
	freq_A = freq_AA + ((1/2) * freq_AB)
	freq_B = freq_BB + ((1/2) * freq_AB)
	return freq_A, freq_B
def hardyWeinberg(df,wgts,effectifs):
	N = len(df)
	freq_A, freq_B = fequenceAlleles(df,wgts)
	obs_effectifs = effectifs
	theo_eff_AA = N * freq_A**2
	theo_eff_AB = N * freq_A * freq_B
	theo_eff_BB = N * freq_B**2
	theo_effectifs = [theo_eff_AA, theo_eff_AB, theo_eff_BB]
	pval = (scipy.stats.chisquare(obs_effectifs, f_exp=theo_effectifs)[1])
	return pval
def desequilibreLiaison(df, wgts):
	freq_A, freq_B = fequenceAlleles(df,wgts)
	DL = 0

################ SIMULATIONS ################
def generateMultivarie(freq_allelique,n,size,df):
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

####### REDEFINE PRIORS #######
def redefinePriors(countGeno):
	maf = 0.28
	nbAA = countGeno[0] * math.sqrt(maf)
	nbAB = 2 * countGeno[1] * maf * (1 - maf)
	nbBB = countGeno[2] * math.sqrt((1 - maf))

if __name__ == '__main__' :
	genotypes=["AA","AB","BB"]
	#list_snp = ["rs6639113","rs17616293","rs9908342","rs34440822","rs34014260","rs7074353","rs163660","rs17567797","rs2518996","rs735476"]
	list_snp = ["rs6639113"]
	#AllDatas = importDatas("/home/e062531t/Bureau/Sidwell/datas/AX-11605013.chp.txt")
	AllDatas = importDatas("/home/e062531t/Bureau/Sidwell/datas/bad_snp_rac/rac.txt")
	AllDatas = AllDatas.sort_values(['snp','Sample'], ascending=[True,False])
	
	for snp in list_snp:
		df_snp = degPlatesFromSNP(AllDatas,snp)
		df_snp = df_snp.reset_index()
		listADN = getADN103(df_snp)
		df_corrected = correct_logRatio_median_interquantile(correct_strength_median(df_snp))
		wgtsGenotypes, meansGenotypes, countGenotypes = getParamsFromDatas(df_corrected,genotypes)
		maf = getMAF_for_snp("/home/e062531t/Bureau/Sidwell/datas/Axiom_GW_Hu_SNP.na34.annot.csv",snp)
		GaussianMixture(df_corrected,wgtsGenotypes,meansGenotypes,genotypes,snp,listADN,maf) # as_matrix -> from df to array
		
		#plotByPlate(df_snp,wgtsGenotypes,meansGenotypes,genotypes,snp)
	
	#generateMultivarie(0.27,96,1, AllDatas)
	#hardyWeinberg(AllDatas,wgtsGenotypes,countGenotypes)

	
	



	



