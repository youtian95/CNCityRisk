class BldMeta():

    NumberOfStories:int = 0
    PlanArea:float = 0
    ReplacementCost:float = 0
    ReplacementTime:float = 0
    StructureType:str = 'UNKNOWN'
    OccupancyClass:str = 'UNKNOWN'
    YearBuilt:int = 0
    Latitude:float = 0
    Longitude:float = 0 

    def __init__(self, **kwargs) -> None:
        for key, val in kwargs.items():
            setattr(self, key, val)