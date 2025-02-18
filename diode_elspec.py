#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

electrodeModels={
	"MDTDirB33005":{
		"full_name": "Medtronic B33005",
		"matfname": "medtronic_b33005",
		"lead_diameter": 1.3,
		"lead_color": 0.7,
		"contact_length": 1.5,
		"contact_diameter": 1.3,
		"contact_color":0.3,
		"tip_diameter":1.3,
		"tip_color":0.7,
		"tip_length":0.9,
		"contact_spacing":0.5,
		"numel":8,
		"tipiscontact":0,
		"markerpos": 15.65,
		"markerlen": 2.3,
		"contactnames": [
			"K0 (R)",
			"K1A (R)",
			"K1B (R)",
			"K1C (R)",
			"K2A (R)",
			"K2B (R)",
			"K2C (R)",
			"K3 (R)",
			"K0 (L)",
			"K1A (L)",
			"K1B (L)",
			"K1C (L)",
			"K2A (L)",
			"K2B (L)",
			"K2C (L)",
			"K3 (L)"
		],
		"isdirected": 1,
		"etagenames":{
			"1":[
				"K0 (R)",
				"K1 (R)",
				"K2 (R)",
				"K3 (R)"
			],
			"2":[
				"K0 (L)",
				"K1 (L)",
				"K2 (L)",
				"K3 (L)"
			]
		},
		"etageidx":[
			(1,1),
			(2,4),
			(5,7),
			(8,8)
		],
		"forstimulation": 1
	},
	"MDTDirB33015":{
		"full_name": "Medtronic B33015",
		"matfname": "medtronic_b33015",
		"lead_diameter": 1.3,
		"lead_color": 0.7,
		"contact_length": 1.5,
		"contact_diameter": 1.3,
		"contact_color":0.3,
		"tip_diameter":1.3,
		"tip_color":0.7,
		"tip_length":0.9,
		"contact_spacing":1.5,
		"numel":8,
		"tipiscontact":0,
		"markerpos": 18.65,
		"markerlen": 2.3,
		"contactnames": [
			"K0 (R)",
			"K1A (R)",
			"K1B (R)",
			"K1C (R)",
			"K2A (R)",
			"K2B (R)",
			"K2C (R)",
			"K3 (R)",
			"K0 (L)",
			"K1A (L)",
			"K1B (L)",
			"K1C (L)",
			"K2A (L)",
			"K2B (L)",
			"K2C (L)",
			"K3 (L)"
		],
		"isdirected": 1,
		"etagenames":{
			"1":[
				"K0 (R)",
				"K1 (R)",
				"K2 (R)",
				"K3 (R)"
			],
			"2":[
				"K0 (L)",
				"K1 (L)",
				"K2 (L)",
				"K3 (L)"
			]
		},
		"etageidx":[
			(1,1),
			(2,4),
			(5,7),
			(8,8)
		],
		"forstimulation": 1
	},
	"BSCDirDB2202":{
		"full_name": "Boston Scientific Vercise Directed",
		"matfname": "boston_vercise_directed",
		"lead_diameter": 1.3,
		"lead_color": 0.7,
		"contact_length": 1.5,
		"contact_diameter": 1.3,
		"contact_color":0.3,
		"tip_diameter":1.3,
		"tip_color":0.3,
		"tip_length":1.5,
		"contact_spacing":0.5,
		"numel":8,
		"tipiscontact": 1,
		"markerpos": 11,
		"markerlen": 3,
		"contactnames": [
			"K9 (R)",
			"K10 (R)",
			"K11 (R)",
			"K12 (R)",
			"K13 (R)",
			"K14 (R)",
			"K15 (R)",
			"K16 (R)",
			"K1 (L)",
			"K2 (L)",
			"K3 (L)",
			"K4 (L)",
			"K5 (L)",
			"K6 (L)",
			"K7 (L)",
			"K8 (L)"
		],
		"isdirected": 1,
		"etagenames":{
			"1":[
				"K9 (R)",
				"K10-12 (R)",
				"K13-15 (R)",
				"K16 (R)"
			],
			"2":[
				"K1 (L)",
				"K2-4 (L)",
				"K5-7 (L)",
				"K8 (L)"
			]
		},
		"etageidx":[
			(1,1),
			(2,4),
			(5,7),
			(8,8)
		],
		"forstimulation": 1
	},
	"SJMDir6172":{
		"full_name": "St. Jude Directed 6172 (short)",
		"matfname": "stjude_directed_05",
		"lead_diameter": 1.27,
		"lead_color": 0.7,
		"contact_length": 1.5,
		"contact_diameter": 1.27,
		"contact_color":0.3,
		"tip_diameter":1.27,
		"tip_color":0.3,
		"tip_length":1.0,
		"contact_spacing":0.5,
		"numel":8,
		"tipiscontact": 0,
		"markerpos": 10.75,
		"markerlen": 1.5,
		"contactnames": [
			"K1 (R)",
			"K2A (R)",
			"K2B (R)",
			"K2C (R)",
			"K3A (R)",
			"K3B (R)",
			"K3C (R)",
			"K4 (R)",
			"K1 (L)",
			"K2A (L)",
			"K2B (L)",
			"K2C (L)",
			"K3A (L)",
			"K3B (L)",
			"K3C (L)",
			"K4 (L)"
		],
		"isdirected": 1,
		"etagenames":{
			"1":[
				"K1 (R)",
				"K2 (R)",
				"K3 (R)",
				"K4 (R)"
			],
			"2":[
				"K1 (L)",
				"K2 (L)",
				"K3 (L)",
				"K4 (L)"
			]
		},
		"etageidx":[
			(1,1),
			(2,4),
			(5,7),
			(8,8)
		],
		"forstimulation": 1
	},
	"SJMDir6173":{
		"full_name": "St. Jude Directed 6173 (long)",
		"matfname": "stjude_directed_15",
		"lead_diameter": 1.27,
		"lead_color": 0.7,
		"contact_length": 1.5,
		"contact_diameter": 1.27,
		"contact_color":0.3,
		"tip_diameter":1.27,
		"tip_color":0.3,
		"tip_length":1.0,
		"contact_spacing":1.5,
		"numel":8,
		"tipiscontact": 0,
		"markerpos": 13.75,
		"markerlen": 1.5,
		"contactnames": [
			"K1 (R)",
			"K2A (R)",
			"K2B (R)",
			"K2C (R)",
			"K3A (R)",
			"K3B (R)",
			"K3C (R)",
			"K4 (R)",
			"K1 (L)",
			"K2A (L)",
			"K2B (L)",
			"K2C (L)",
			"K3A (L)",
			"K3B (L)",
			"K3C (L)",
			"K4 (L)"
		],
		"isdirected": 1,
		"etagenames":{
			"1":[
				"K1 (R)",
				"K2 (R)",
				"K3 (R)",
				"K4 (R)"
			],
			"2":[
				"K1 (L)",
				"K2 (L)",
				"K3 (L)",
				"K4 (L)"
			]
		},
		"etageidx":[
			(1,1),
			(2,4),
			(5,7),
			(8,8)
		],
		"forstimulation": 1
	}
}

# json_output = json.dumps(electrodeModels, indent=4)
# with open('/home/greydon/Documents/GitHub/DiODE_python/diode_elspec.json', 'w') as fid:
# 	fid.write(json_output)
# 	fid.write('\n')
			