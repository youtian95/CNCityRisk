import BldCluster as bc

BldDirtFile = 'Resources\SanFrancisco_buildings_full.csv'
obj = bc.BldCluster(BldDirtFile)
obj.ClassifyBld(IgnoredLabels=['id','Latitude','Longitude','YearBuilt','ReplacementCost'], 
    **{'PlanArea': 100})