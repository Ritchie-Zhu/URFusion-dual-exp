"""Epoch-level checks for ConflictAdd_1.

Flags interpret the gate. They do not stop training.
After eval, GATE_OFF or (GATE_ON and GATE_FLAT) without beating S1 and A2
means this is not a MoE contribution.
"""


def health_flags(row):
	flags = []
	a_mean = float(row['a_mean'])
	a_std = float(row['a_std'])
	vis_u = float(row['vis_u_l1'])
	ir_u = float(row['ir_u_l1'])
	e_base = float(row['e_base_l1'])
	res_conflict = float(row['res_conflict'])

	if a_mean < 0.05:
		flags.append('GATE_OFF')
	if a_mean > 0.95:
		flags.append('GATE_ON')
	if a_std < 0.02:
		flags.append('GATE_FLAT')
	if vis_u < 1e-3 and ir_u < 1e-3:
		flags.append('UNIQUE_DEAD')
	if e_base < 1e-6:
		flags.append('BASE_DEAD')
	if e_base > 1e-8 and res_conflict < 0.01 * e_base:
		flags.append('RESIDUAL_DEAD')
	if not flags:
		return 'OK'
	return ','.join(flags)


def format_health_line(row):
	return (
		f"[HEALTH] {row['health']} | "
		f"a_mean={row['a_mean']:.3f} a_std={row['a_std']:.3f} "
		f"a_vs_mask={row.get('a_vs_mask', 0):.3f} "
		f"mask={row['mask_mean']:.3f}±{row.get('mask_std', 0):.3f} | "
		f"res={row['res_conflict']:.4f} e_conflict={row['e_conflict_l1']:.4f} "
		f"e_base={row['e_base_l1']:.4f} | "
		f"vis_u={row['vis_u_l1']:.4f} ir_u={row['ir_u_l1']:.4f} cap={row['cap_l1']:.4f} "
		f"diff={row.get('diff_l1', 0):.4f}"
	)
