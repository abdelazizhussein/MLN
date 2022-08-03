import os 
import pandas as pd
import json
from Loader.scifiTracks import ScifiTrack
import matplotlib.pyplot as plt
def DataLoader(json_path:str,model_config:str,features):
    """
    Loads data from Json file into a pandas dataframe
    
    """
    assert os.path.isfile(json_path), "Json file cannot be found, please check path"
    with open(json_path, "r") as f:
        events = json.load(f)
        filteredGhosts = 0
        filteredGood = 0
        totGhosts = 0
        totTracks = 0

        scifitracks = []
        for event in events:
            # matching:
            matches = []
            for detector, tracks in event.items():
                for track in tracks:
                    #prepare scifi tracks
                    if detector == "scifi":
                        scifi = ScifiTrack(track)
                        attributes = vars(scifi)
                        scifitracks.append(attributes)
                        #simple filter of requiring  a track to have at least 10 hits                            
                        if scifi.numHits() > 10:
                            if scifi.isGhost: filteredGhosts += 1
                            else: filteredGood += 1
        
                        if scifi.isGhost: totGhosts += 1
                        totTracks += 1
                    #prepare velo tracks
                    elif detector == "velo":
                        continue

                    #match velo and scifi tracks
                    elif detector=="matches":
                        continue
        dataset = pd.DataFrame.from_dict(scifitracks)
        labels = dataset['isGhost']
        dataset=dataset[features]

        #scale dataset to be between -1 and 1
        dataset = 2*(dataset - dataset.min())/(dataset.max() - dataset.min()) -1
        
        #print out information about the number of ghost tracks present in the dataset
        print("Ghosts fraction: "+str(totGhosts)+" / "+str(totTracks)+" ("+str(100*totGhosts/totTracks)+"%)")
        #Ghosts eliminated from a simple cut on the number of hits
        print("Ghosts passed: "+str(filteredGhosts)+" / "+str(totGhosts)+" ("+str(100*filteredGhosts/totGhosts)+"%)")
        print("Tracks passed: "+str(filteredGood)+" / "+str(totTracks-totGhosts)+" ("+str(100*filteredGood/(totTracks-totGhosts))+"%)")
        print("New Ghosts fraction: "+str(filteredGhosts)+" / "+str(filteredGood)+" ("+str(100*filteredGhosts/filteredGood)+"%)")
        return dataset,labels
