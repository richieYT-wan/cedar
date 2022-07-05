import pandas as pd
import imm_overlap_tools
import datetime
import random


def mapIDX2prot(clust, idx2protDict):
    return [set(clust), set([idx2protDict[c] for c in clust])]


def getIDXprotClustList(df, IDXs, protCol='UID'):
    idx2protDict = dict(zip(range(len(df)), df[protCol].values))
    clustList = [mapIDX2prot(clust, idx2protDict) for clust in IDXs]
    return sorted(clustList, key=lambda x: len(x[1]), reverse=True)


def getKmerClust_prot(clustList, k=9):
    setList = []
    for i, (idxs, prots) in enumerate(clustList):
        overlapFlag = False
        for j in range(len(setList)):
            if len(setList[j][1].intersection(prots)) > 0:
                setList[j][0] = setList[j][0].union(idxs)
                setList[j][1] = setList[j][1].union(prots)
                overlapFlag = True
                break
        if not overlapFlag:
            setList.append([idxs, prots])
    IDXs, Prots = zip(*setList)
    return [list(idx) for idx in IDXs]


def getKmerClust(peptides, k=9):
    setList = []
    for i, pept in enumerate(peptides):
        kmerSet = imm_overlap_tools.kmerSet(pept, k=k)
        overlapFlag = False
        for j in range(len(setList)):
            if len(setList[j][0].intersection(kmerSet)) > 0:
                setList[j][0] = setList[j][0].union(kmerSet)
                setList[j][1].append(i)
                overlapFlag = True
                break
        if not overlapFlag:
            setList.append([kmerSet, [i]])
    kmerSets, IDXs = zip(*setList)
    return IDXs


def makePartitions(idxList, parts=5):
    partitions = [[] for i in range(parts)]
    idxList = sorted(idxList, key=lambda x: len(x))
    while len(idxList) > 0:
        idxs = idxList.pop()
        partitions[0] += idxs
        partitions = sorted(partitions, key=lambda x: len(x))
    return partitions


def CVpartitionsIDXs(idxs, fold=5):
    testIDXs = idxs
    trainIDXs = [[] for x in range(fold)]
    for i in range(fold):
        for j, idx in enumerate(idxs):
            if j != i:
                trainIDXs[i] += idx
        # trainIDXs.append([k for k in idxs for j,idx in enumerate(idxs) if j!=i])
    return testIDXs, trainIDXs


def getPartitionColumn(testIDXs):
    partitions = [0] * sum([len(i) for i in testIDXs])
    for i, parts in enumerate(testIDXs):
        print(i, len(parts))
        for j, idx in enumerate(parts):
            partitions[idx] = i
    return partitions


def commonMotifPartitionWrapper(df, fold=5, k=9, peptCol='Peptide', addPartCol=True):
    IDXs = getKmerClust(df[peptCol].values, k=k)
    partitions = makePartitions(IDXs, parts=fold)
    testIDXs, trainIDXs = CVpartitionsIDXs(partitions, fold=fold)
    if addPartCol:
        df['Partition'] = getPartitionColumn(testIDXs)
    return testIDXs, trainIDXs, df


def commonMotifPartitionWrapper_Prot(df, fold=5, k=9, peptCol='Peptide', protCol='UID', addPartCol=True):
    print("Start - {}".format(datetime.datetime.now().strftime("%H:%M:%S")))
    IDXsPept = getKmerClust(df[peptCol].values, k=k)
    print("Common Motif Clustering - {}".format(datetime.datetime.now().strftime("%H:%M:%S")))
    clustList = getIDXprotClustList(df, IDXsPept, protCol=protCol)
    print("Map IDX to Prot - {}".format(datetime.datetime.now().strftime("%H:%M:%S")))
    IDXs = getKmerClust_prot(clustList)
    print("Protein Level Clustering - {}".format(datetime.datetime.now().strftime("%H:%M:%S")))
    partitions = makePartitions(IDXs, parts=fold)
    print("Partitions Made - {}".format(datetime.datetime.now().strftime("%H:%M:%S")))
    testIDXs, trainIDXs = CVpartitionsIDXs(partitions, fold=fold)
    print("CV folds made - {}".format(datetime.datetime.now().strftime("%H:%M:%S")))
    if addPartCol:
        df['Partition'] = getPartitionColumn(testIDXs)
    return testIDXs, trainIDXs, df


def genRandomGroupPartitions(df, groupCol='Patient', fold=5):
    df = df.copy(deep=True)
    groups = df[groupCol].unique()
    numGroup = len(groups)
    groups = sorted(groups)
    random.seed(42)
    partitions = random.choices(range(fold), k=numGroup)
    groupPartitionMapper = dict(zip(groups, partitions))
    df['Partition'] = df[groupCol].map(groupPartitionMapper)
    return df


def motifPatientClusterPositivesAddNegatives(df, fold=5, targetCol='Target', peptCol='PeptMut', patientCol='Patient'):
    print(len(df))
    # Cluster positives by common motifs and patients
    dfPos = df[df[targetCol] == 1.0]
    dfNeg = df[df[targetCol] == 0.0]
    print(len(dfPos) + len(dfNeg))
    testIDXs, trainIDXs, dfPosPartition = commonMotifPartitionWrapper_Prot(dfPos, fold=fold, k=8, peptCol=peptCol,
                                                                           protCol=patientCol, addPartCol=True)

    # Assign partition and add negative peptides to corresponding positives
    dfList = []
    posPats = []
    for (partition, patient), dfG in dfPosPartition.groupby(['Partition', patientCol]):
        dfNegPat = dfNeg[dfNeg[patientCol] == patient].copy(deep=True)
        dfNegPat['Partition'] = partition
        dfList.append(dfNegPat)
        posPats.append(patient)
    dfPosNeg = pd.concat([pd.concat(dfList), dfPosPartition])
    print(len(dfPosNeg))

    # Randomly assign cluster to patients with no positive peptide data
    dfNegNoPos = dfNeg[dfNeg[patientCol].apply(lambda pat: not pat in posPats)]
    dfNegNoPos = genRandomGroupPartitions(dfNegNoPos, fold=fold)
    print(len(dfNegNoPos))
    dfOut = pd.concat([dfPosNeg, dfNegNoPos])
    print(len(dfOut))

    return dfOut