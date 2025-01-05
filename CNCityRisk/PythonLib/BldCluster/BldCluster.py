########################################################
# Classify buildings according to their parameters (which can be
# plan area, structural type, number of stories, occupancy type,
# and seismic design leve). There is a representative building 
# for each cluster of buildings.
# 
# 3rd party dependancy: 
# - pandas, tqdm, numpy
########################################################

from typing import *
import pandas as pd
import numpy as np
from tqdm import tqdm

class BldCluster():

    ReprBld: pd.DataFrame = pd.DataFrame() # parameters of rep blds
    OriginalBld: pd.DataFrame = pd.DataFrame()
    Bld2ReprBld: pd.DataFrame = pd.DataFrame() # maps original bld ID to rep bld ID

    def __init__(self, BldDirectoryFile: str, nrows = None) -> None:
        self.OriginalBld = pd.read_csv(BldDirectoryFile, index_col=None, header=0, nrows=nrows)
        self.OriginalBld = self.OriginalBld.fillna('unknown')

    def ClassifyBld(self, IgnoredLabels=['id','Latitude','Longitude'], TargetLabels = None, **labels: Dict[str,float]):
        '''
        Classify buildings based on their properties, including structural type, occupancy type, design level, etc.
        Usage:
            ClassifyBld(IgnoredLabels=['id','Latitude','Longitude'], TargetLabels = None, **labels). 
            ClassifyBld(IgnoredLabels=None, TargetLabels = ['area','story'...], **labels)
        Parameters:
            labels (dict). labels and increments that are used for classification.
                StructureType or OccupancyClass can't be the selected labels.
            IgnoredLabels (list). labels that are ignored during classification
            TargetLabels (list). labels that are used for classification
        Returns:    
            NumOfGroups (dict). Number of groups for each label
        '''

        # label group boundary
        LabelGroup_Mid: Dict[str,np.array[float]] = {}
        LabelGroups_lower: Dict[str,np.array[float]] = {}
        for label,incr in labels.items():
            datacol = np.array(self.OriginalBld[label].values)
            max_val = np.ceil(datacol.max() / incr) * incr
            min_val = np.floor(datacol.min() / incr) * incr
            Num_lab = np.ceil((max_val - min_val)/incr)
            min_val= (max_val + min_val)/2.0 - Num_lab*incr/2.0
            LabelGroups_lower[label] = np.linspace(min_val, min_val+incr*(Num_lab-1), int(Num_lab))
            LabelGroup_Mid[label] = LabelGroups_lower[label] + incr/2.0
            if np.issubdtype(self.OriginalBld[label].dtype, np.integer):
                LabelGroup_Mid[label] = np.rint(LabelGroup_Mid[label])

        # target headers
        if TargetLabels is not None and IgnoredLabels is not None:
            raise Exception('TargetLabels and IgnoredLabels can not be both specified')
        elif TargetLabels is not None:
            headersname:List[str] = TargetLabels
        elif IgnoredLabels is not None:
            headersname:List[str] = self.OriginalBld.columns.values.tolist()
            for label in IgnoredLabels:
                headersname.remove(label)
        else:
            raise Exception('TargetLabels and IgnoredLabels can not be both None')

        # construct intermediate dataframe, where the labels are replaced by the representative values
        # InterMdf = pd.DataFrame()
        # for ind, row in tqdm(self.OriginalBld.iterrows()):
        #     for label,incr in labels.items():
        #         # representative value
        #         iloc = np.argmin(np.abs(row[label]-LabelGroup_Mid[label]))
        #         repval = LabelGroup_Mid[label][iloc]
        #         row[label] = repval
        #     row_new = row[headersname]
        #     InterMdf = pd.concat([InterMdf, row_new])
        
        # construct intermediate dataframe, where the labels are replaced by the representative values
        InterMdf: pd.DataFrame = self.OriginalBld.copy(deep=True)
        InterMdf = InterMdf[headersname]
        def classify_row(row, labels, LabelGroup_Mid):
            for label,incr in labels.items():
                # representative value
                iloc = np.argmin(np.abs(row[label]-LabelGroup_Mid[label]))
                repval = LabelGroup_Mid[label][iloc]
                row[label] = repval
            return row
        InterMdf = InterMdf.apply(classify_row, args=(labels, LabelGroup_Mid), axis=1)

        
        self.ReprBld = pd.DataFrame(columns=headersname)
        self.Bld2ReprBld = pd.DataFrame({'Original Buildings index':InterMdf.index.tolist(), 
            'Representative Buildings index':[0]*InterMdf.index.size})

        # generate representative blds
        group = InterMdf.groupby(InterMdf.columns.values.tolist())
        i = 0
        for key, value in group:
            self.ReprBld = pd.concat([self.ReprBld, pd.DataFrame(value.iloc[0]).transpose()], ignore_index=True)
            self.Bld2ReprBld.loc[value.index.values.tolist(),['Representative Buildings index']] = i
            i = i+1
        
        self.ReprBld.to_csv('Representative Buildings.csv')
        self.Bld2ReprBld.to_csv('Bld2ReprBld.csv')



        


    
            

        