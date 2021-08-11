import silver_spectacle as ss

data = {
    "labels": [
        8,
        2,
        4,
        0,
        6,
        1,
        9,
        5,
        3,
        7
    ],
    "datasets": [
        {
            "label": "split",
            "backgroundColor": "rgb( 0,  92, 192, 0.9)",
            "borderColor": "rgb( 0,  92, 192, 0.9)",
            "data": [
                9770,
                9862,
                9886,
                9919,
                9940,
                9933,
                9838,
                9834,
                9913,
                9919
            ]
        },
        {
            "label": "simple",
            "backgroundColor": "rgb(75, 192, 192, 0.9)",
            "borderColor": "rgb(75, 192, 192, 0.9)",
            "data": [
                9651,
                9778,
                9815,
                9880,
                9895,
                9916,
                9744,
                9831,
                9791,
                9866
            ]
        },
        {
            "label": "fresh",
            "backgroundColor": "rgb(0, 292, 192, 0.9)",
            "borderColor": "rgb(0, 292, 192, 0.9)",
            "data": [
                9690,
                9807,
                9856,
                9843,
                9875,
                9857,
                9480,
                9771,
                9670,
                9767
            ]
        }
    ]
}

ss.DisplayCard("chartjs", {
    "type": 'line',
    "options": {
        "pointRadius": 3, # the size of the dots
        "scales": {
            "y": {
                "min": 9700,
                "max": 10000,
            },
        }
    },
    "data": data,
})

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