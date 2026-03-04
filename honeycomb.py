import tidy3d as td 
import numpy as np 

def create_centered_honeycomb_slab(
    lattice_const: float = 1.0,     # 晶格常数 a (um)
    disk_diameter: float = 0.45,     # 圆柱直径 (um)
    slab_thickness: float = 0.22,   # 圆柱长度/厚度 (um)
    twist_angle: float = 0.0,       # 旋转角度 (度)
    domain_size: tuple = (50.0, 50.0), # (Lx, Ly) 生成矩形区域的大小 (um)
    material: td.Medium = td.Medium(permittivity=3.47**2), # 硅
    centerz: float = 0.0,
):
    """
    生成一个以 (0,0,0) 为六边形空心中心的蜂窝状光子晶体板，
    占据矩形区域，并支持旋转。
    """
    
    radius = disk_diameter / 2.0
    Lx, Ly = domain_size
    
    # 1. 定义三角晶格的基矢量 (a1, a2)
    # a1 沿 x 轴
    a1 = np.array([lattice_const, 0])
    a2 = np.array([lattice_const * 0.5, lattice_const * np.sqrt(3) / 2])

    # 2. 定义基元 (Basis) 并计算"空心化"偏移量
    # 标准定义下，原子位于 (0,0) 和 (a1+a2)/3 (即 a/2, a*sqrt(3)/6)
    # 在这种定义下，(0,0) 是一个原子。
    # 蜂窝结构的"空心"中心位于该原子上方距离 a/sqrt(3) 的位置（沿 y 轴，如果晶格取向是标准的）
    # 为了让 (0,0) 成为空心，我们需要把整个晶格向下平移。
    # 几何计算：偏移量应为 (0, -a / sqrt(3))
    
    shift_vector = np.array([0, -lattice_const / np.sqrt(3)])
    
    # 原始基元位置
    basis_1_raw = np.array([0, 0])
    basis_2_raw = np.array([lattice_const * 0.5, lattice_const * np.sqrt(3) / 6.0])
    
    # 应用偏移，使得 (0,0) 处没有原子，且周围原子呈六重对称
    basis_1 = basis_1_raw + shift_vector
    basis_2 = basis_2_raw + shift_vector

    # 3. 预计算旋转矩阵
    theta = np.radians(twist_angle)
    c, s = np.cos(theta), np.sin(theta)
    rot_matrix = np.array([[c, -s], 
                           [s, c]])

    structures_raw = []

    # 4. 生成网格点
    # 扩大搜索范围以确保旋转后矩形角落也能被填满
    search_radius = np.sqrt((Lx/2)**2 + (Ly/2)**2)
    n_max = int(search_radius / lattice_const) + 3
    
    for i in range(-n_max, n_max + 1):
        for j in range(-n_max, n_max + 1):
            # 晶胞原点
            cell_origin = i * a1 + j * a2
            
            # 两个原子的位置（在未旋转的坐标系中，且已对其中心化）
            points_raw = [cell_origin + basis_1, cell_origin + basis_2]
            
            for p in points_raw:
                # 5. 应用旋转 (先旋转，再判断是否在矩形内)
                # 这样模拟的是：在一个无限大的旋转晶格上，切下来一块矩形芯片
                p_rot = rot_matrix @ p
                
                # 6. 矩形区域筛选 (Rectangular Crop)
                # 检查 x 和 y 是否在设定范围内
                if (abs(p_rot[0]) <= Lx / 2.0) and (abs(p_rot[1]) <= Ly / 2.0):
                    
                    # 7. 创建 Tidy3D 结构
                    # 使用 length 替代 height
                    cyl = td.Cylinder(
                        radius=radius,
                        length=slab_thickness,
                        axis=2, # z轴
                        center=(p_rot[0], p_rot[1], centerz)
                    )
                    
                    structures_raw.append(cyl)
                
    return structures_raw