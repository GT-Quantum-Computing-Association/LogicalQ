phase_flip_code = {
    "label": (3,1,2),
    "stabilizer_tableau": [
        "XXI",
        "IXX",
    ],
}
four_qubit_code = {
    "label": (4,2,2),
    "stabilizer_tableau": [
        "XZZX",
        "YZZY",
    ],
}

five_qubit_code = {
    "label": (5,1,3),
    "stabilizer_tableau": [
        "XZZXI",
        "IXZZX",
        "XIXZZ",
        "ZXIXZ",
    ],
}

steane_code = {
    "label": (7,1,3),
    "stabilizer_tableau": [
      "XXXXIII",
      "IXXIXXI",
      "IIXXIXX",
      "ZZZZIII",
      "IZZIZZI",
      "IIZZIZZ",
    ]
}

eight_qubit_code = {
    "label": (8,3,3),
    "stabilizer_tableau": [
        "XXXXXXXX",
        "ZZZZZZZZ",
        "IXIXYZYZ",
        "IXZYIXZY",
        "IYXZXZIY",
    ],
},

stabilizer_codes = [
    phase_flip_code,
    four_qubit_code,
    five_qubit_code,
    steane_code,
    eight_qubit_code
]
