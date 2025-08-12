# Quantum Error-Correcting Codes

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
    eight_qubit_code,
]

implemented_codes = [
    five_qubit_code,
    steane_code,
]

# Mapping from QECC labels to objects
# NOTE - Use the robust label_to_qecc function in code
qecc_dict = {
    (5,1,3): five_qubit_code,
    (7,1,3): steane_code,
}

# Utility functions

def label_to_qecc(_qecc_label):
    qecc_label = tuple(_qecc_label)
    if qecc_label in qecc_dict.keys():
        return qecc_dict[qecc_label]
    else:
        raise KeyError(f"Input {_qecc_label} is not a valid QEC label.")

