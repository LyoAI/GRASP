import matplotlib.pyplot as plt

# Data from the provided text file
gradient_data = {
    'model.layers.0': {'self_attn.q_proj': 70.1443, 'self_attn.k_proj': 132.4030, 'self_attn.v_proj': 2113.4409, 'self_attn.o_proj': 3951.6560,
                       'mlp.gate_proj': 7787.7217, 'mlp.up_proj': 12240.0674, 'mlp.down_proj': 13656.8945},
    'model.layers.1': {'self_attn.q_proj': 633.9106, 'self_attn.k_proj': 293.2051, 'self_attn.v_proj': 2111.1128, 'self_attn.o_proj': 5645.1611,
                       'mlp.gate_proj': 7109.4844, 'mlp.up_proj': 12025.9668, 'mlp.down_proj': 12788.4648},
    'model.layers.2': {'self_attn.q_proj': 904.4398, 'self_attn.k_proj': 290.3190, 'self_attn.v_proj': 1758.4368, 'self_attn.o_proj': 3778.4463,
                       'mlp.gate_proj': 7261.6357, 'mlp.up_proj': 11744.9453, 'mlp.down_proj': 15273.5732},
    'model.layers.3': {'self_attn.q_proj': 1146.5393, 'self_attn.k_proj': 390.0372, 'self_attn.v_proj': 1550.5100, 'self_attn.o_proj': 3769.9927,
                       'mlp.gate_proj': 7755.6519, 'mlp.up_proj': 13623.3477, 'mlp.down_proj': 15378.2559},
    'model.layers.4': {'self_attn.q_proj': 1026.8757, 'self_attn.k_proj': 500.8434, 'self_attn.v_proj': 1923.1747, 'self_attn.o_proj': 4347.4863,
                       'mlp.gate_proj': 6968.3086, 'mlp.up_proj': 12184.6641, 'mlp.down_proj': 12959.5879},
    'model.layers.5': {'self_attn.q_proj': 1225.5813, 'self_attn.k_proj': 616.4426, 'self_attn.v_proj': 1760.8055, 'self_attn.o_proj': 5119.3306,
                       'mlp.gate_proj': 5222.9414, 'mlp.up_proj': 12516.4600, 'mlp.down_proj': 13873.1865},
    'model.layers.6': {'self_attn.q_proj': 1056.0269, 'self_attn.k_proj': 408.2621, 'self_attn.v_proj': 1365.6937, 'self_attn.o_proj': 4121.4634,
                       'mlp.gate_proj': 5809.0444, 'mlp.up_proj': 11862.8057, 'mlp.down_proj': 12293.6113},
    'model.layers.7': {'self_attn.q_proj': 1415.7231, 'self_attn.k_proj': 504.1701, 'self_attn.v_proj': 1304.3029, 'self_attn.o_proj': 4163.1445,
                       'mlp.gate_proj': 6045.7368, 'mlp.up_proj': 12242.9697, 'mlp.down_proj': 12221.0645},
    'model.layers.8': {'self_attn.q_proj': 1145.1633, 'self_attn.k_proj': 585.0316, 'self_attn.v_proj': 2228.5720, 'self_attn.o_proj': 5917.5215,
                       'mlp.gate_proj': 6074.3555, 'mlp.up_proj': 11335.3379, 'mlp.down_proj': 11886.2627},
    'model.layers.9': {'self_attn.q_proj': 1642.4773, 'self_attn.k_proj': 578.7480, 'self_attn.v_proj': 1425.7380, 'self_attn.o_proj': 5398.8994,
                       'mlp.gate_proj': 5845.3911, 'mlp.up_proj': 11403.1973, 'mlp.down_proj': 11507.1143},
    'model.layers.10': {'self_attn.q_proj': 1654.6635, 'self_attn.k_proj': 742.8400, 'self_attn.v_proj': 1782.8798, 'self_attn.o_proj': 8568.9043,
                        'mlp.gate_proj': 5849.5586, 'mlp.up_proj': 10968.0264, 'mlp.down_proj': 12801.8896},
    'model.layers.11': {'self_attn.q_proj': 1486.4506, 'self_attn.k_proj': 496.4076, 'self_attn.v_proj': 1736.9182, 'self_attn.o_proj': 6269.6230,
                        'mlp.gate_proj': 8127.1895, 'mlp.up_proj': 16147.1992, 'mlp.down_proj': 14571.9961},
    'model.layers.12': {'self_attn.q_proj': 1256.3916, 'self_attn.k_proj': 1011.4996, 'self_attn.v_proj': 3123.9932, 'self_attn.o_proj': 7022.3945,
                        'mlp.gate_proj': 5258.5732, 'mlp.up_proj': 10051.7422, 'mlp.down_proj': 10653.2207},
    'model.layers.13': {'self_attn.q_proj': 1212.7285, 'self_attn.k_proj': 473.0164, 'self_attn.v_proj': 1524.4209, 'self_attn.o_proj': 5407.5776,
                        'mlp.gate_proj': 4949.1387, 'mlp.up_proj': 9535.2061, 'mlp.down_proj': 10055.9346},
    'model.layers.14': {'self_attn.q_proj': 1084.9824, 'self_attn.k_proj': 458.1448, 'self_attn.v_proj': 1312.8884, 'self_attn.o_proj': 6234.3730,
                        'mlp.gate_proj': 4610.0293, 'mlp.up_proj': 8707.5840, 'mlp.down_proj': 11598.6191},
    'model.layers.15': {'self_attn.q_proj': 1334.0032, 'self_attn.k_proj': 702.9846, 'self_attn.v_proj': 1684.1316, 'self_attn.o_proj': 6369.2422,
                        'mlp.gate_proj': 5285.1519, 'mlp.up_proj': 10085.5234, 'mlp.down_proj': 13767.0127},
    'model.layers.16': {'self_attn.q_proj': 987.3105, 'self_attn.k_proj': 447.5149, 'self_attn.v_proj': 2098.1182, 'self_attn.o_proj': 2492.1152,
                        'mlp.gate_proj': 5547.7100, 'mlp.up_proj': 10565.8525, 'mlp.down_proj': 15307.7490},
    'model.layers.17': {'self_attn.q_proj': 965.2249, 'self_attn.k_proj': 1128.2629, 'self_attn.v_proj': 3063.9839, 'self_attn.o_proj': 3375.2793,
                        'mlp.gate_proj': 3933.0195, 'mlp.up_proj': 7843.7988, 'mlp.down_proj': 9268.9707},
    'model.layers.18': {'self_attn.q_proj': 955.7485, 'self_attn.k_proj': 495.3810, 'self_attn.v_proj': 528.6908, 'self_attn.o_proj': 1637.9346,
                        'mlp.gate_proj': 3923.7803, 'mlp.up_proj': 6468.7471, 'mlp.down_proj': 7893.7158},
    'model.layers.19': {'self_attn.q_proj': 1322.0332, 'self_attn.k_proj': 711.7699, 'self_attn.v_proj': 1334.5701, 'self_attn.o_proj': 3240.8596,
                        'mlp.gate_proj': 4468.4141, 'mlp.up_proj': 7739.2197, 'mlp.down_proj': 9781.8125},
    'model.layers.20': {'self_attn.q_proj': 709.9998, 'self_attn.k_proj': 521.1989, 'self_attn.v_proj': 1125.3348, 'self_attn.o_proj': 2927.4978,
                        'mlp.gate_proj': 4692.0693, 'mlp.up_proj': 7084.4092, 'mlp.down_proj': 10086.7031},
    'model.layers.21': {'self_attn.q_proj': 615.5952, 'self_attn.k_proj': 451.6238, 'self_attn.v_proj': 553.3666, 'self_attn.o_proj': 3021.9673,
                        'mlp.gate_proj': 4133.8892, 'mlp.up_proj': 5296.4434, 'mlp.down_proj': 10364.0244},
    'model.layers.22': {'self_attn.q_proj': 618.4153, 'self_attn.k_proj': 528.7681, 'self_attn.v_proj': 717.6113, 'self_attn.o_proj': 2953.3159,
                        'mlp.gate_proj': 4742.0225, 'mlp.up_proj': 6361.8145, 'mlp.down_proj': 8549.9854},
    'model.layers.23': {'self_attn.q_proj': 527.6690, 'self_attn.k_proj': 333.7169, 'self_attn.v_proj': 1659.1726, 'self_attn.o_proj': 3608.5776,
                        'mlp.gate_proj': 6497.8779, 'mlp.up_proj': 9527.6250, 'mlp.down_proj': 13439.3926}
}

# Preparing the plot again with the extended data
layers = list(gradient_data.keys())
weight_types = list(gradient_data[layers[0]].keys())

# Plot each weight type as a separate line
for weight_type in weight_types:
    values = [gradient_data[layer][weight_type] for layer in layers]
    plt.plot(layers, values, label=weight_type)

# Customize the plot
plt.xlabel('Layer ID')
plt.ylabel('Gradient Value')
plt.title('Gradient Distribution Across Layers (Extended)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()
