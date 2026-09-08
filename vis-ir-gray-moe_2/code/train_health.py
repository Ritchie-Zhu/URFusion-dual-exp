"""Epoch-level checks for leak-free spatial residual MoE."""


def health_flags(row):
	flags = []
	g_mean = float(row['g_mean'])
	g_std = float(row['g_std'])
	vis_u = float(row['vis_u_l1'])
	ir_u = float(row['ir_u_l1'])
	res_vis = float(row['res_vis'])
	res_ir = float(row['res_ir'])
	e_base = float(row['e_base_l1'])
	res_sum = res_vis + res_ir

	if g_mean < 0.05 or g_mean > 0.95:
		flags.append('GATE_COLLAPSE')
	if g_std < 0.02:
		flags.append('GATE_FLAT')
	if vis_u < 1e-3 and ir_u < 1e-3:
		flags.append('UNIQUE_DEAD')
	# Unique has nowhere else to go; dead residuals mean unique never enters fusion.
	if e_base > 1e-8 and res_sum < 0.01 * e_base:
		flags.append('RESIDUAL_DEAD')
	if not flags:
		return 'OK'
	return ','.join(flags)


def format_health_line(row):
	return (
		f"[HEALTH] {row['health']} | "
		f"g_mean={row['g_mean']:.3f} g_std={row['g_std']:.3f} "
		f"mask={row['mask_mean']:.3f}±{row.get('mask_std', 0):.3f} | "
		f"res_vis={row['res_vis']:.4f} res_ir={row['res_ir']:.4f} e_base={row['e_base_l1']:.4f} | "
		f"vis_u={row['vis_u_l1']:.4f} ir_u={row['ir_u_l1']:.4f} cap={row['cap_l1']:.4f} "
		f"diff={row.get('diff_l1', 0):.4f}"
	)
