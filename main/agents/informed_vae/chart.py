import silver_spectacle as ss

# convert simple+fresh to model=fresh
# filter out everything other than testing
# filter out everything with non zero error number
# average across experiment number, within model name, within binary_class_order, within 
# create dataset for each model name

data = {
    "type": "line",
    "options": {
        "pointRadius": 3,
        "scales": {
            "y": {
                "min": 9700,
                "max": 10000
            }
        }
    },
    "data": {
        "labels": [0,1,2,3,4,5,6,7,8,9],
        "datasets": [
            {
                "label": "split",
                "backgroundColor": "rgb( 0,  92, 192, 0.9)",
                "borderColor": "rgb( 0,  92, 192, 0.9)",
                "data": [
                    9755,
                    9923,
                    9841,
                    9879
                ]
            },
            {
                "label": "simple",
                "backgroundColor": "rgb(75, 192, 192, 0.9)",
                "borderColor": "rgb(75, 192, 192, 0.9)",
                "data": [
                    9682,
                    9774,
                    9755,
                    9871
                ]
            },
            {
                "label": "fresh",
                "backgroundColor": "rgb(0, 292, 192, 0.9)",
                "borderColor": "rgb(0, 292, 192, 0.9)",
                "data": [
                    9714,
                    9799,
                    9721,
                    9842
                ]
            }
        ]
    }
}

ss.DisplayCard("chartjs", data)

# ss.DisplayCard("chartjs", {
#     "type": 'line',
#     "options": {
#         "pointRadius": 3, # the size of the dots
#         "scales": {
#             "y": {
#                 "min": 9700,
#                 "max": 10000,
#             },
#         }
#     },
#     "data": {
#         "labels": [9,8,3,],
#         "datasets": [
#             {
#                 "label": "split1",
#                 "data": [
#                     9697,
#                     9783,
#                     9844,
#                 ],
#                 # "fill": True,
#                 "tension": 0.1,
#                 "backgroundColor": 'rgb(0, 92, 192, 0.9)',
#                 "borderColor":'rgb(0, 92, 192, 0.9)',
#             },
#             {
#                 "label": "split2",
#                 "data": [
#                     9760,
#                     9792,
#                     9830,
#                 ],
#                 # "fill": True,
#                 "tension": 0.1,
#                 "backgroundColor": 'rgb(0, 92, 192, 0.9)',
#                 "borderColor":'rgb(0, 92, 192, 0.9)',
#             },
#             {
#                 "label": "split3",
#                 "data": [
#                     9760,
#                     9840,
#                     9912,
#                 ],
#                 # "fill": True,
#                 "tension": 0.1,
#                 "backgroundColor": 'rgb(0, 92, 192, 0.9)',
#                 "borderColor":'rgb(0, 92, 192, 0.9)',
#             },
#             {
#                 "label": "simple1",
#                 "data": [
#                     9793,
#                     9836,
#                     9887,
#                 ],
#                 # "fill": True,
#                 "tension": 0.1,
#                 "backgroundColor": 'rgb(75, 192, 192, 0.9)',
#                 "borderColor":'rgb(75, 192, 192, 0.9)',
#             },
#             {
#                 "label": "simple2",
#                 "data": [
#                     9831,
#                     9911,
#                     9872,
#                 ],
#                 # "fill": True,
#                 "tension": 0.1,
#                 "backgroundColor": 'rgb(75, 192, 192, 0.9)',
#                 "borderColor":'rgb(75, 192, 192, 0.9)',
#             },
#             {
#                 "label": "simple3",
#                 "data": [
#                     9831,
#                     9868,
#                     9895,
#                 ],
#                 # "fill": True,
#                 "tension": 0.1,
#                 "backgroundColor": 'rgb(75, 192, 192, 0.9)',
#                 "borderColor":'rgb(75, 192, 192, 0.9)',
#             },
#             {
#                 "label": "fresh1",
#                 "data": [
#                     9825,
#                     9786,
#                     9791,
#                 ],
#                 # "fill": True,
#                 "tension": 0.1,
#                 "backgroundColor": 'rgb(0, 292, 192, 0.9)',
#                 "borderColor":'rgb(0, 292, 192, 0.9)',
#             },
#             {
#                 "label": "fresh2",
#                 "data": [
#                     9795,
#                     9826,
#                     9826,
#                 ],
#                 # "fill": True,
#                 "tension": 0.1,
#                 "backgroundColor": 'rgb(0, 292, 192, 0.9)',
#                 "borderColor":'rgb(0, 292, 192, 0.9)',
#             },
#             {
#                 "label": "fresh3",
#                 "data": [
#                     9849,
#                     9770,
#                     9897,
#                 ],
#                 # "fill": True,
#                 "tension": 0.1,
#                 "backgroundColor": 'rgb(0, 292, 192, 0.9)',
#                 "borderColor":'rgb(0, 292, 192, 0.9)',
#             },
#         ]
#     },
# })