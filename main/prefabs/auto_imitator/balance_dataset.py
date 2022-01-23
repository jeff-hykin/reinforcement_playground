from super_map import Map
from informative_iterator import ProgressBar
from tools.agent_recorder import AgentRecorder

number_of_metadata_entries = 2848404
database = AgentRecorder(save_to="resources/datasets.ignore/atari/baselines_pretrained@breakout_custom")

def percentize(frequency: Map):
    percents = Map()
    total = sum(frequency[Map.Values])
    for key, value in frequency:
        percents[key] = value/total
    return percents
    
frequency = Map()
for progress, metadata in ProgressBar(database.load_metadata(), iterations=database.size):
    frequency[metadata] += 1
    if progress.updated:
        print(str(percentize(frequency)))

print('frequency = ', frequency)



# percents:
# {
#     1: 0.33906955502796565, 
#     0: 0.19401303833769104, 
#     3: 0.274277602051202, 
#     2: 0.19263980458314128, 
# }