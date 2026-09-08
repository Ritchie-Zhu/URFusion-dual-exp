"""S1 health: no router, so no GATE_* flags."""


def health_flags(row):
	flags = []
	vis_u = float(row['vis_u_l1'])
	ir_u = float(row['ir_u_l1'])
	e_base = float(row['e_base_l1'])
	if vis_u < 1e-3 and ir_u < 1e-3:
		flags.append('UNIQUE_DEAD')
	if e_base < 1e-6:
		flags.append('BASE_DEAD')
	if not flags:
		return 'OK'
	return ','.join(flags)


def format_health_line(row):
	return (
		f"[HEALTH] {row['health']} | "
		f"e_base={row['e_base_l1']:.4f} | "
		f"vis_u={row['vis_u_l1']:.4f} ir_u={row['ir_u_l1']:.4f} cap={row['cap_l1']:.4f} | "
		f"mask={row['mask_mean']:.3f}±{row.get('mask_std', 0):.3f} (log only, not in loss)"
	)
