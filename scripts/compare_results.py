"""
比较史瓦西黑洞和非奇异黑洞的通量分布差异
计算定量比较结果用于论文
"""
import os
import numpy as np
import json
import matplotlib.pyplot as plt

def load_flux_data(file_path):
    """加载通量数据"""
    if not os.path.exists(file_path):
        return None
    data = np.load(file_path)
    return data["b"], data["alpha"], data["F"]

def extract_psi0(filename):
    """从文件名提取psi0值"""
    import re
    match = re.search(r'psi0=([0-9.]+)', filename)
    return float(match.group(1)) if match else None

def calculate_peak_flux(b, F, tolerance=0.1):
    """计算峰值通量及其位置"""
    if len(F) == 0:
        return None, None
    max_idx = np.argmax(F)
    peak_b = b[max_idx]
    peak_F = F[max_idx]
    return peak_b, peak_F

def calculate_flux_width(b, F, fraction=0.5):
    """计算通量分布的半高全宽（FWHM）"""
    if len(F) == 0:
        return None
    max_F = np.max(F)
    half_max = max_F * fraction
    indices = np.where(F >= half_max)[0]
    if len(indices) == 0:
        return None
    width = b[indices[-1]] - b[indices[0]]
    return width

def compare_flux_profiles(schwarzschild_dir, nonsingular_dir, output_dir):
    """比较两种黑洞的通量分布"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 读取配置
    config_path = os.path.join(base_dir, "config", "config.json")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    r_in = config["r_in"]
    theta0 = config["theta0_deg"]
    kappaff = config["kappa_ff"]
    kappaK = config["kappa_K"]
    opt_regime = config["optical_regime"]
    r_max = config["r_max"]
    dalpha = config["dalpha"]
    tolerance = dalpha / 2
    
    psi0_list = [5.0, 30.0, 60.0, 70.0, 90.0]
    
    results = []
    
    print("=" * 80)
    print("Comparison of Schwarzschild and Nonsingular Black Hole Flux Profiles")
    print("=" * 80)
    print(f"\nParameters: r_in={r_in}, theta0={theta0}°, kappa_ff={kappaff}, kappa_K={kappaK}")
    print(f"Optical regime: {opt_regime}")
    print("\n" + "-" * 80)
    
    for psi0_deg in psi0_list:
        filename = f"flux_rmax={r_max:.1f}_optical_{opt_regime}_psi0={psi0_deg:.1f}_rin={r_in:.1f}_theta0={theta0:.1f}_kappaff={kappaff:.3f}_kappaK={kappaK:.3f}.npz"
        
        sch_file = os.path.join(schwarzschild_dir, filename)
        nons_file = os.path.join(nonsingular_dir, filename)
        
        if not os.path.exists(sch_file) or not os.path.exists(nons_file):
            print(f"\npsi0 = {psi0_deg:.1f} deg: Missing data files, skipping...")
            continue
        
        # 加载数据
        b_sch, alpha_sch, F_sch = load_flux_data(sch_file)
        b_nons, alpha_nons, F_nons = load_flux_data(nons_file)
        
        # Y轴方向分析 (α ≈ 0, π)
        mask_sch_y = (np.abs(alpha_sch - 0.0) < tolerance) | (np.abs(alpha_sch - np.pi) < tolerance)
        mask_nons_y = (np.abs(alpha_nons - 0.0) < tolerance) | (np.abs(alpha_nons - np.pi) < tolerance)
        
        b_sch_y = b_sch[mask_sch_y]
        F_sch_y = F_sch[mask_sch_y] * 1e5
        b_nons_y = b_nons[mask_nons_y]
        F_nons_y = F_nons[mask_nons_y] * 1e5
        
        # 计算峰值
        peak_b_sch_y, peak_F_sch_y = calculate_peak_flux(b_sch_y, F_sch_y)
        peak_b_nons_y, peak_F_nons_y = calculate_peak_flux(b_nons_y, F_nons_y)
        
        if peak_F_sch_y is None or peak_F_nons_y is None:
            continue
        
        # 计算差异
        peak_diff_y = abs(peak_F_nons_y - peak_F_sch_y)
        peak_diff_percent_y = (peak_diff_y / peak_F_sch_y) * 100
        
        # 计算宽度
        width_sch_y = calculate_flux_width(b_sch_y, F_sch_y)
        width_nons_y = calculate_flux_width(b_nons_y, F_nons_y)
        
        if width_sch_y is not None and width_nons_y is not None:
            width_diff_y = abs(width_nons_y - width_sch_y)
            width_diff_percent_y = (width_diff_y / width_sch_y) * 100
        else:
            width_diff_y = None
            width_diff_percent_y = None
        
        # Z轴方向分析 (α ≈ π/2, 3π/2)
        mask_sch_z = (np.abs(alpha_sch - np.pi/2) < tolerance) | (np.abs(alpha_sch - 3*np.pi/2) < tolerance)
        mask_nons_z = (np.abs(alpha_nons - np.pi/2) < tolerance) | (np.abs(alpha_nons - 3*np.pi/2) < tolerance)
        
        b_sch_z = b_sch[mask_sch_z]
        F_sch_z = F_sch[mask_sch_z] * 1e5
        b_nons_z = b_nons[mask_nons_z]
        F_nons_z = F_nons[mask_nons_z] * 1e5
        
        peak_b_sch_z, peak_F_sch_z = calculate_peak_flux(b_sch_z, F_sch_z)
        peak_b_nons_z, peak_F_nons_z = calculate_peak_flux(b_nons_z, F_nons_z)
        
        if peak_F_sch_z is not None and peak_F_nons_z is not None:
            peak_diff_z = abs(peak_F_nons_z - peak_F_sch_z)
            peak_diff_percent_z = (peak_diff_z / peak_F_sch_z) * 100
        else:
            peak_diff_z = None
            peak_diff_percent_z = None
        
        width_sch_z = calculate_flux_width(b_sch_z, F_sch_z)
        width_nons_z = calculate_flux_width(b_nons_z, F_nons_z)
        
        if width_sch_z is not None and width_nons_z is not None:
            width_diff_z = abs(width_nons_z - width_sch_z)
            width_diff_percent_z = (width_diff_z / width_sch_z) * 100
        else:
            width_diff_z = None
            width_diff_percent_z = None
        
        # 保存结果
        result = {
            'psi0': psi0_deg,
            'Y_axis': {
                'peak_b_sch': peak_b_sch_y,
                'peak_F_sch': peak_F_sch_y,
                'peak_b_nons': peak_b_nons_y,
                'peak_F_nons': peak_F_nons_y,
                'peak_diff_percent': peak_diff_percent_y,
                'width_sch': width_sch_y,
                'width_nons': width_nons_y,
                'width_diff_percent': width_diff_percent_y
            },
            'Z_axis': {
                'peak_b_sch': peak_b_sch_z,
                'peak_F_sch': peak_F_sch_z,
                'peak_b_nons': peak_b_nons_z,
                'peak_F_nons': peak_F_nons_z,
                'peak_diff_percent': peak_diff_percent_z,
                'width_sch': width_sch_z,
                'width_nons': width_nons_z,
                'width_diff_percent': width_diff_percent_z
            }
        }
        results.append(result)
        
        # 打印结果
        print(f"\npsi0 = {psi0_deg:.1f} deg:")
        print(f"  Y-axis (alpha = 0, pi):")
        print(f"    Peak flux - Schwarzschild: {peak_F_sch_y:.4f} at b = {peak_b_sch_y:.4f}M")
        print(f"    Peak flux - Nonsingular:   {peak_F_nons_y:.4f} at b = {peak_b_nons_y:.4f}M")
        print(f"    Relative difference:        {peak_diff_percent_y:.2f}%")
        if width_diff_percent_y is not None:
            print(f"    Width difference:           {width_diff_percent_y:.2f}%")
        
        if peak_diff_percent_z is not None:
            print(f"  Z-axis (alpha = pi/2, 3pi/2):")
            print(f"    Peak flux - Schwarzschild: {peak_F_sch_z:.4f} at b = {peak_b_sch_z:.4f}M")
            print(f"    Peak flux - Nonsingular:   {peak_F_nons_z:.4f} at b = {peak_b_nons_z:.4f}M")
            print(f"    Relative difference:        {peak_diff_percent_z:.2f}%")
            if width_diff_percent_z is not None:
                print(f"    Width difference:           {width_diff_percent_z:.2f}%")
    
    # 找出差异最大的条件
    if len(results) > 0:
        print("\n" + "=" * 80)
        print("Summary: Conditions with Maximum Differences")
        print("=" * 80)
        
        max_diff_y = max([r['Y_axis']['peak_diff_percent'] for r in results])
        max_diff_z_vals = [r['Z_axis']['peak_diff_percent'] for r in results if r['Z_axis']['peak_diff_percent'] is not None]
        max_diff_z = max(max_diff_z_vals) if len(max_diff_z_vals) > 0 else None
        
        for r in results:
            if r['Y_axis']['peak_diff_percent'] == max_diff_y:
                print(f"\nMaximum Y-axis difference: {max_diff_y:.2f}% at psi0 = {r['psi0']:.1f} deg")
            if max_diff_z is not None and r['Z_axis']['peak_diff_percent'] is not None and r['Z_axis']['peak_diff_percent'] == max_diff_z:
                print(f"Maximum Z-axis difference: {max_diff_z:.2f}% at psi0 = {r['psi0']:.1f} deg")
    else:
        print("\nNo comparison results available. Please ensure both Schwarzschild and Nonsingular data files exist.")
    
    # 保存结果到文件
    output_file = os.path.join(output_dir, "comparison_results.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Comparison Results\n")
        f.write("=" * 80 + "\n\n")
        for r in results:
            f.write(f"psi0 = {r['psi0']:.1f} deg:\n")
            f.write(f"  Y-axis peak difference: {r['Y_axis']['peak_diff_percent']:.2f}%\n")
            if r['Z_axis']['peak_diff_percent'] is not None:
                f.write(f"  Z-axis peak difference: {r['Z_axis']['peak_diff_percent']:.2f}%\n")
            f.write("\n")
    
    print(f"\nResults saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 使用相对路径
    schwarzschild_output = os.path.join(base_dir, "output")
    nonsingular_base = os.path.join(os.path.dirname(base_dir), "nonsingular_Shadow_TD_v2.1_Eng")
    nonsingular_output = os.path.join(nonsingular_base, "output")
    output_dir = os.path.join(base_dir, "output")
    
    print(f"Schwarzschild output dir: {schwarzschild_output}")
    print(f"Nonsingular output dir: {nonsingular_output}")
    print(f"Output dir: {output_dir}\n")
    
    results = compare_flux_profiles(schwarzschild_output, nonsingular_output, output_dir)
