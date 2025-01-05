########################################################
# Python tools for cmd 
# 
# Usage:
# 
# 
# dependency: 
# - pandas, numpy, json
########################################################

import argparse
import sys
import json

import BldCluster as bc


def run(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--BldDirtFile',type=str)
    parser.add_argument('--IgnoredLabels',nargs='+',type=str)
    parser.add_argument('--IncrLabels',type=str)
    args = parser.parse_args(args)

    obj = bc.BldCluster(args.BldDirtFile)
    obj.ClassifyBld(IgnoredLabels=args.IgnoredLabels, **json.loads(args.IncrLabels))


# BldDirtFile = 'E:\CityResilienceAndResilientStructure\BuildingData\SanFrancisco_buildings_partial.csv'
# ignoredLabels=['id','Latitude','Longitude','YearBuilt','ReplacementCost','Footprint']
# IncrLabels = '{"PlanArea": 100}'
# run(['--BldDirtFile',BldDirtFile,'--IgnoredLabels', *ignoredLabels, '--IncrLabels',IncrLabels])


if __name__ == "__main__":
    run(sys.argv[1:])

