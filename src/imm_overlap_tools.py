import os
import pandas as pd
import numpy as np


def kmerize(pept, k=9):
    if len(pept) < k:
        raise Exception('Peptide: {} shorter than k: {}, Exiting!'.format(pept, k))
    else:
        return [pept[i:i + k] for i in range(len(pept) - k + 1)]


def kmerSet(pept, k=9):
    return set(kmerize(pept, k=k))


def makeRefset(pepts, k=9):
    refSet = set()
    for pept in pepts:
        qSet = kmerSet(pept, k=k)
        refSet = refSet.union(qSet)
    return refSet


def readEpiFiles(datDir, filename, k=9):
    refSet = set()
    with open(os.path.join(datDir, filename), 'r') as fh:
        for line in fh:
            line = line.strip()
            qSet = kmerSet(line, k=k)
            refSet = refSet.union(qSet)
    return refSet


def removeOverlap(pept, refSet, k=9):
    qSet = kmerSet(pept, k=k)
    if len(qSet.intersection(refSet)) == 0:
        return True
    else:
        return False


def findOverlap(df, refSet, peptCol='Peptide', k=9):
    return df[~df[peptCol].apply(lambda pept: removeOverlap(pept, refSet, k=k))]


def findNonOverlap(df, refSet, peptCol='Peptide', k=9):
    return df[df[peptCol].apply(lambda pept: removeOverlap(pept, refSet, k=k))]


def getOverlapHeatmap(df, catCol='Dataset', k=8, peptCol='PeptMut'):
    categories = df[catCol].unique()
    numCat = len(categories)
    overlapMat = np.zeros((numCat, numCat))
    overlapMatRel = np.zeros((numCat, numCat))
    for i, cat1 in enumerate(categories):
        for j, cat2 in enumerate(categories):
            df1 = df[df[catCol] == cat1]
            df2 = df[df[catCol] == cat2]
            refSet = makeRefset(df2[peptCol].values, k=k)
            df1_overlap = findOverlap(df1, refSet, peptCol=peptCol, k=k)
            overlapMat[i, j] = len(df1_overlap)
            overlapMatRel[i, j] = len(df1_overlap) / len(df1)
    return categories, overlapMat, overlapMatRel
