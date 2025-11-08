import numpy as np
from math import cos, sin, radians
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


# =========================================================================
# 1. OBJ файл
# =========================================================================

class OBJModel:
    """Класс для работы с OBJ моделями на основе Polyhedron"""

    def __init__(self, polyhedron=None):
        self.polyhedron = polyhedron

    def load_from_file(self, filename):
        """Загрузка модели из OBJ файла"""
        try:
            vertices = []
            faces = []

            with open(filename, 'r') as file:
                for line in file:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    parts = line.split()
                    if not parts:
                        continue

                    if parts[0] == 'v':  # вершина
                        if len(parts) >= 4:
                            vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                            vertices.append(vertex)

                    elif parts[0] == 'f':  # грань
                        face = []
                        for part in parts[1:]:
                            # Обработка формата vertex/texture/normal
                            vertex_index = part.split('/')[0]
                            if vertex_index:
                                face.append(int(vertex_index) - 1)  # OBJ использует 1-based индексацию
                        if len(face) >= 3:
                            faces.append(face)

            self.polyhedron = Polyhedron(vertices, faces)
            return True

        except Exception as e:
            print(f"Ошибка загрузки файла: {e}")
            return False

    def save_to_file(self, filename):
        """Сохранение модели в OBJ файл"""
        try:
            if self.polyhedron is None:
                return False

            with open(filename, 'w') as file:
                file.write("# OBJ файл\n")
                file.write("# Создан программой визуализации многогранников\n")

                # Запись вершин
                # Безопасное деление с проверкой на ноль
                w = self.polyhedron.V[3, :]
                vertices = np.zeros((3, self.polyhedron.V.shape[1]))
                for i in range(self.polyhedron.V.shape[1]):
                    if abs(w[i]) > 1e-10:
                        vertices[:, i] = self.polyhedron.V[:3, i] / w[i]
                    else:
                        vertices[:, i] = self.polyhedron.V[:3, i]
                
                for i in range(vertices.shape[1]):
                    vertex = vertices[:, i]
                    file.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")

                # Запись граней
                for face in self.polyhedron.faces:
                    face_line = "f"
                    for vertex_index in face.indices:
                        face_line += f" {vertex_index + 1}"  # OBJ использует 1-based индексацию
                    file.write(face_line + "\n")

            return True

        except Exception as e:
            print(f"Ошибка сохранения файла: {e}")
            return False



def to_h(point3):
    """Возвращает однородный 4x1 вектор из 3D точки (x, y, z)."""
    x, y, z = point3
    return np.array([x, y, z, 1.0], dtype=float)

def from_h(vec4):
    """Возвращает 3D точку из однородного вектора после перспективного деления."""
    w = vec4[3]
    if w == 0:
        raise ZeroDivisionError("Однородная координата w == 0 при дегомогенизации")
    return (vec4[:3] / w)

def normalize(v):
    """Нормализует вектор."""
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n

# --------------------
# Матрицы преобразований (4x4, вектор-столбцы)
# --------------------

def T(dx, dy, dz):
    """Матрица переноса (смещения)."""
    M = np.eye(4)
    M[:3, 3] = [dx, dy, dz]
    return M

def S(sx, sy, sz):
    """Матрица масштабирования."""
    M = np.eye(4)
    M[0,0], M[1,1], M[2,2] = sx, sy, sz
    return M

def Rx(angle_deg):
    """Матрица поворота вокруг оси X."""
    a = radians(angle_deg)
    ca, sa = cos(a), sin(a)
    M = np.eye(4)
    M[1,1], M[1,2] = ca, -sa
    M[2,1], M[2,2] = sa,  ca
    return M

def Ry(angle_deg):
    """Матрица поворота вокруг оси Y."""
    a = radians(angle_deg)
    ca, sa = cos(a), sin(a)
    M = np.eye(4)
    M[0,0], M[0,2] =  ca, sa
    M[2,0], M[2,2] = -sa, ca
    return M

def Rz(angle_deg):
    """Матрица поворота вокруг оси Z."""
    a = radians(angle_deg)
    ca, sa = cos(a), sin(a)
    M = np.eye(4)
    M[0,0], M[0,1] = ca, -sa
    M[1,0], M[1,1] = sa,  ca
    return M

def reflect(plane: str):
    """Отражение относительно координатной плоскости: 'xy', 'yz', или 'xz'."""
    plane = plane.lower()
    if plane == "xy":
        return S(1, 1, -1)
    if plane == "yz":
        return S(-1, 1, 1)
    if plane == "xz":
        return S(1, -1, 1)
    raise ValueError("Плоскость должна быть: 'xy', 'yz', 'xz'")

def rodrigues_axis_angle(axis, angle_deg):
    """Поворот 3x3 вокруг единичной оси на угол в градусах."""
    axis = normalize(np.asarray(axis, dtype=float))
    a = radians(angle_deg)
    c, s = cos(a), sin(a)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]], dtype=float)
    R = np.eye(3)*c + (1-c)*np.outer(axis, axis) + s*K
    return R

def R_around_line(p1, p2, angle_deg):
    """Матрица поворота 4x4 вокруг произвольной 3D линии p1->p2 на угол."""
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    axis = p2 - p1
    R3 = rodrigues_axis_angle(axis, angle_deg)  # 3x3
    M = np.eye(4)
    M[:3,:3] = R3
    return T(*p1) @ M @ T(*(-p1))

# --------------------
# Матрицы проекций
# --------------------

def perspective(f=1.5):
    """
    Простая матрица перспективной проекции.
    Камера в начале координат смотрит вдоль +Z; точки сцены должны иметь z > 0.
    """
    # Важно: хотим w' = z/f (а не 1 + z/f). Для этого элемент (3,3) должен быть 0.
    # Тогда после дегомогенизации x' = f*x/z, y' = f*y/z.
    M = np.eye(4)
    M[3,3] = 0.0
    M[3,2] = 1.0 / f  # w' = z/f  -> x' = x / (z/f) = f*x/z
    return M

# --------------------
# Класс камеры
# --------------------

class Camera:
    """
    Класс камеры для 3D визуализации.
    
    Камера задается:
    - position: положение камеры в мировых координатах
    - target: точка, на которую смотрит камера
    - up: вектор "вверх" камеры
    - fov: поле зрения (field of view) в градусах
    - aspect: соотношение сторон (ширина/высота)
    - near, far: ближняя и дальняя плоскости отсечения
    """
    
    def __init__(self, position=None, target=None, up=None, fov=60.0, aspect=1.0, near=0.1, far=100.0):
        self.position = np.array(position if position is not None else [0.0, 0.0, 5.0])
        self.target = np.array(target if target is not None else [0.0, 0.0, 0.0])
        self.up = np.array(up if up is not None else [0.0, 1.0, 0.0])
        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far
        
        # Параметры орбитального вращения
        self.orbit_radius = np.linalg.norm(self.position - self.target)
        self.orbit_theta = 0.0  # угол по горизонтали (азимут)
        self.orbit_phi = 0.0    # угол по вертикали (высота)
        self._update_orbit_angles()
    
    def _update_orbit_angles(self):
        """Обновляет углы орбиты на основе текущей позиции."""
        rel_pos = self.position - self.target
        self.orbit_radius = np.linalg.norm(rel_pos)
        if self.orbit_radius > 0:
            rel_pos_norm = rel_pos / self.orbit_radius
            # phi - угол от горизонтальной плоскости
            self.orbit_phi = np.arcsin(np.clip(rel_pos_norm[1], -1.0, 1.0))
            # theta - угол в горизонтальной плоскости
            self.orbit_theta = np.arctan2(rel_pos_norm[0], rel_pos_norm[2])
    
    def get_view_direction(self):
        """Возвращает нормализованный вектор направления обзора (от камеры к цели)."""
        direction = self.target - self.position
        return normalize(direction)
    
    def get_view_matrix(self):
        """
        Возвращает матрицу вида (view matrix), которая преобразует мировые координаты
        в координаты камеры.
        """
        # Вектор направления (от камеры к цели)
        forward = normalize(self.target - self.position)
        # Правый вектор
        right = normalize(np.cross(forward, self.up))
        # Истинный вектор вверх (перпендикулярен forward и right)
        true_up = np.cross(right, forward)
        
        # Матрица вращения (переход в систему координат камеры)
        R = np.eye(4)
        R[0, :3] = -right    # Инвертируем X для правильной визуализации
        R[1, :3] = -true_up  # Инвертируем Y для правильной визуализации
        R[2, :3] = -forward  # В системе камеры смотрим вдоль -Z
        
        # Матрица переноса (перемещение мира так, чтобы камера была в начале координат)
        T_mat = T(-self.position[0], -self.position[1], -self.position[2])
        
        # View matrix = R * T
        return R @ T_mat
    
    def get_projection_matrix(self):
        """
        Возвращает матрицу перспективной проекции.
        Использует упрощенную модель с фокусным расстоянием.
        """
        # Вычисляем фокусное расстояние из угла обзора
        f = 1.0 / np.tan(np.radians(self.fov / 2.0))
        return perspective(f)
    
    def orbit_rotate(self, delta_theta, delta_phi):
        """
        Вращает камеру вокруг целевой точки (орбитальное вращение).
        
        delta_theta: изменение азимутального угла (в градусах)
        delta_phi: изменение угла высоты (в градусах)
        """
        self.orbit_theta += np.radians(delta_theta)
        self.orbit_phi += np.radians(delta_phi)
        
        # Ограничиваем угол phi, чтобы избежать переворота
        max_phi = np.radians(89.0)
        self.orbit_phi = np.clip(self.orbit_phi, -max_phi, max_phi)
        
        # Вычисляем новую позицию на сфере
        x = self.orbit_radius * np.cos(self.orbit_phi) * np.sin(self.orbit_theta)
        y = self.orbit_radius * np.sin(self.orbit_phi)
        z = self.orbit_radius * np.cos(self.orbit_phi) * np.cos(self.orbit_theta)
        
        self.position = self.target + np.array([x, y, z])
    
    def zoom(self, delta):
        """
        Приближает/отдаляет камеру от цели.
        
        delta: изменение расстояния (положительное - приближение, отрицательное - отдаление)
        """
        self.orbit_radius = max(0.5, self.orbit_radius - delta)
        
        # Обновляем позицию, сохраняя направление
        direction = normalize(self.position - self.target)
        self.position = self.target + direction * self.orbit_radius
    
    def set_target(self, target):
        """Устанавливает новую целевую точку."""
        self.target = np.array(target)
        self._update_orbit_angles()
    
    def reset(self, position=None, target=None):
        """Сбрасывает камеру в исходное положение."""
        if position is not None:
            self.position = np.array(position)
        if target is not None:
            self.target = np.array(target)
        self._update_orbit_angles()

def ortho_xy():
    """Ортографическая проекция на плоскость XY (отбрасывание Z)."""
    M = np.eye(4)
    M[2,2] = 0.0
    return M

def isometric_projection_matrix():
    """Аксонометрическая (изометрическая) проекция = поворот + ортографическая проекция."""
    # Классическая изометрия: поворот вокруг Z на 45°, затем вокруг X на ~35.264°
    alpha = 35.264389682754654
    beta = 45.0
    R = Rx(alpha) @ Rz(beta)
    return ortho_xy() @ R

# --------------------
# Геометрические классы
# --------------------

class Point:
    """Класс для представления точки в 3D пространстве."""
    def __init__(self, x, y, z):
        self.v = to_h((x, y, z))

    @property
    def xyz(self):
        return from_h(self.v)

    def as_array(self):
        return self.v.copy()

class PolygonFace:
    """Класс для представления грани многогранника."""
    def __init__(self, vertex_indices):
        self.indices = list(vertex_indices)
        self.normal = None  # Вектор нормали грани
    
    def compute_normal(self, vertices, object_center=None):
        """
        Вычисляет вектор нормали грани через векторное произведение.
        Гарантирует, что нормаль направлена наружу от центра объекта.
        
        vertices: массив 3xN координат вершин (уже в 3D, без однородной координаты)
        object_center: центр объекта (для определения направления наружу)
        """
        if len(self.indices) < 3:
            self.normal = np.array([0.0, 0.0, 1.0])
            return
        
        # Берем первые три вершины грани
        p0 = vertices[:, self.indices[0]]
        p1 = vertices[:, self.indices[1]]
        p2 = vertices[:, self.indices[2]]
        
        # Два вектора в плоскости грани
        v1 = p1 - p0
        v2 = p2 - p0
        
        # Векторное произведение дает вектор нормали
        normal = np.cross(v1, v2)
        
        # Нормализуем
        norm_length = np.linalg.norm(normal)
        if norm_length > 1e-10:
            normal = normal / norm_length
        else:
            self.normal = np.array([0.0, 0.0, 1.0])
            return
        
        # Проверяем, что нормаль направлена наружу от центра объекта
        if object_center is not None:
            # Центр грани
            face_center = np.mean(vertices[:, self.indices], axis=1)
            # Вектор от центра объекта к центру грани
            outward = face_center - object_center
            
            # Если нормаль направлена внутрь (скалярное произведение < 0), переворачиваем её
            if np.dot(normal, outward) < 0:
                normal = -normal
        
        self.normal = normal

class Polyhedron:
    """Класс для представления многогранника."""
    def __init__(self, vertices, faces):
        """
        vertices: список вершин (кортежи (x,y,z))
        faces: список граней (списки индексов вершин)
        """
        self.V = np.array([to_h(p) for p in vertices], dtype=float).T  # 4xN (столбец = вершина)
        self.faces = [PolygonFace(f) for f in faces]
        self.compute_face_normals()
    
    def compute_face_normals(self):
        """Вычисляет нормали для всех граней многогранника."""
        # Получаем 3D координаты вершин (без однородной координаты)
        vertices_3d = self.V[:3, :] / self.V[3, :]
        
        # Центр объекта
        center = np.mean(vertices_3d, axis=1)
        
        for face in self.faces:
            face.compute_normal(vertices_3d, center)

    # --- основные методы ---
    def copy(self):
        """Создает копию многогранника."""
        P = Polyhedron([(0,0,0)], [[]])
        P.V = self.V.copy()
        P.faces = [PolygonFace(f.indices.copy()) for f in self.faces]
        P.compute_face_normals()
        return P

    def center(self):
        """Вычисляет центр многогранника."""
        pts = self.V[:3, :] / self.V[3, :]
        return np.mean(pts, axis=1)

    def apply(self, M):
        """Применяет матричное преобразование 4x4."""
        self.V = M @ self.V
        self.compute_face_normals()  # Пересчитываем нормали после преобразования
        return self

    # --- удобные методы преобразований (все через матрицы) ---
    def translate(self, dx, dy, dz):
        """Перенос (смещение)."""
        return self.apply(T(dx, dy, dz))

    def scale(self, sx, sy, sz):
        """Масштабирование."""
        return self.apply(S(sx, sy, sz))

    def scale_about_center(self, s):
        c = self.center()
        return self.apply(T(*(-c)) @ S(s, s, s) @ T(*c))

    def rotate_x(self, angle_deg):
        """Поворот вокруг оси X."""
        return self.apply(Rx(angle_deg))

    def rotate_y(self, angle_deg):
        """Поворот вокруг оси Y."""
        return self.apply(Ry(angle_deg))

    def rotate_z(self, angle_deg):
        """Поворот вокруг оси Z."""
        return self.apply(Rz(angle_deg))

    def reflect(self, plane: str):
        """Отражение относительно координатной плоскости."""
        return self.apply(reflect(plane))

    def rotate_around_axis_through_center(self, axis: str, angle_deg):
        axis = axis.lower()
        c = self.center()
        R = {'x': Rx, 'y': Ry, 'z': Rz}[axis](angle_deg)
        return self.apply(T(*(-c)) @ R @ T(*c))

    def rotate_around_line(self, p1, p2, angle_deg):
        """Поворот вокруг произвольной линии."""
        return self.apply(R_around_line(p1, p2, angle_deg))

    # --- вспомогательные методы для ребер ---
    def edges(self):
        """Вычисляет список ребер многогранника."""
        es = set()
        if self.faces and len(self.faces[0].indices)>0:
            # Строим ребра из граней
            for f in self.faces:
                idx = f.indices
                for i in range(len(idx)):
                    a = idx[i]
                    b = idx[(i+1) % len(idx)]
                    es.add(tuple(sorted((a,b))))
        else:
            # Резервный метод: соединение ближайших соседей
            pts = (self.V[:3,:] / self.V[3,:]).T
            n = len(pts)
            D = np.linalg.norm(pts[None,:,:]-pts[:,None,:], axis=-1)
            for i in range(n):
                neigh = list(np.argsort(D[i])[1:4])  # 3 ближайших соседа
                for j in neigh:
                    es.add(tuple(sorted((i,j))))
        return sorted(list(es))

    # --- проекция ---
    def projected(self, matrix4x4):
        """Возвращает 2D точки (x,y) после применения матрицы проекции."""
        Pv = matrix4x4 @ self.V
        # Перспективное деление
        Pv = Pv / Pv[3, :]
        # возвращаем только (x,y)
        return Pv[0, :], Pv[1, :]

# --------------------
# Правильные многогранники (Платоновы тела)
# --------------------

def tetrahedron():
    """Создает правильный тетраэдр с центром в начале координат."""
    V = [(1, 1, 1),
         (1,-1,-1),
         (-1, 1,-1),
         (-1,-1, 1)]
    F = [
        [0,1,2],
        [0,3,1],
        [0,2,3],
        [1,3,2]
    ]
    return Polyhedron(V, F)

def hexahedron():
    """Правильный гексаэдр (куб) с центром в начале координат и ребром 2."""
    V = [
        (-1, -1, -1),  # 0
        ( 1, -1, -1),  # 1
        ( 1,  1, -1),  # 2
        (-1,  1, -1),  # 3
        (-1, -1,  1),  # 4
        ( 1, -1,  1),  # 5
        ( 1,  1,  1),  # 6
        (-1,  1,  1),  # 7
    ]
    # 6 квадратных граней (порядок вершин по контуру)
    F = [
        [0, 1, 2, 3],  # z = -1 (низ)
        [4, 5, 6, 7],  # z = +1 (верх)
        [0, 1, 5, 4],  # y = -1
        [1, 2, 6, 5],  # x = +1
        [2, 3, 7, 6],  # y = +1
        [3, 0, 4, 7],  # x = -1
    ]
    return Polyhedron(V, F)

def octahedron():
    """Правильный октаэдр с центром в начале координат и ребром √2."""
    V = [
        ( 1,  0,  0),  # 0
        (-1,  0,  0),  # 1
        ( 0,  1,  0),  # 2
        ( 0, -1,  0),  # 3
        ( 0,  0,  1),  # 4 (верх)
        ( 0,  0, -1),  # 5 (низ)
    ]
    # 8 треугольных граней
    F = [
        [4, 0, 2],
        [4, 2, 1],
        [4, 1, 3],
        [4, 3, 0],
        [5, 2, 0],
        [5, 1, 2],
        [5, 3, 1],
        [5, 0, 3],
    ]
    return Polyhedron(V, F)

def icosahedron():
    """Икосаэдр, построенный с цилиндра.
    Полюса: (0,0,±sqrt(5)/2); кольца радиуса 1 на z=±1/2, нижнее смещено на 36°.
    Возвращает Polyhedron с явными 20 треугольными гранями.
    """
    def deg(a): return np.deg2rad(a)

    z_top, z_bot = +0.5, -0.5
    r = 1.0
    z_pole = np.sqrt(5.0) / 2.0

    V = []
    # верхняя вершина
    V.append((0.0, 0.0, +z_pole))

    # 1..5 — верхнее кольцо (углы 0,72,144,216,288)
    for k in range(5):
        ang = deg(72*k)
        V.append((r*np.cos(ang), r*np.sin(ang), z_top))

    # 6..10 — нижнее кольцо (углы 36,108,180,252,324)
    for k in range(5):
        ang = deg(36 + 72*k)
        V.append((r*np.cos(ang), r*np.sin(ang), z_bot))

    # нижняя вершина
    V.append((0.0, 0.0, -z_pole))

    F = []

    # Верхняя «шапка»: 5 треугольников (0, Ti, Ti+1)
    for i in range(5):
        F.append([0, 1+i, 1+((i+1) % 5)])

    # Средняя зона: 10 треугольников (по 2 на «сектор»).
    # Важный момент: у вершины верхнего кольца Ti ближайшие нижние — Bi и B(i-1).
    for i in range(5):
        Ti   = 1 + i
        Tip1 = 1 + ((i+1) % 5)
        Bi   = 6 + i
        Bim1 = 6 + ((i-1) % 5)

        # «верхний» из пары (Ti, Bi, B(i-1))
        F.append([Ti, Bi, Bim1])
        # «нижний» из пары (Bi, Tip1, Ti)
        F.append([Bi, Tip1, Ti])

    # Нижняя «шапка»: 5 треугольников (11, Bj+1, Bj)
    for j in range(5):
        Bj   = 6 + j
        Bjp1 = 6 + ((j+1) % 5)
        F.append([11, Bjp1, Bj])

    return Polyhedron(V, F)


def dodecahedron():
    """Додекаэдр как дуал к идущему выше 'цилиндрическому' икосаэдру:
    вершины = центры тяжести треугольных граней икосаэдра,
    грани = пятиугольники, по одному на каждую вершину икосаэдра.
    """
    I = icosahedron()
    # координаты вершин икосаэдра (N x 3)
    V = (I.V[:3, :] / I.V[3, :]).T #normalization
    faces_I = [f.indices for f in I.faces]  # список где каждый элемент это список индексов точек грани

    # 20 вершин додекаэдра: центроиды треугольников
    D_vertices = [tuple(np.mean(V[idxs], axis=0)) for idxs in faces_I]

    # каждой вершине икосаэдра ставим в соответствие номера граней в которых она используется
    incident = [[] for _ in range(len(V))]
    for fi, tri in enumerate(faces_I):
        for vid in tri:
            incident[vid].append(fi)

    # Построим 12 пятиугольных граней додекаэдра.
    D_faces = []
    for vid, fids in enumerate(incident):
        if len(fids) != 5:
            # на всякий случай пропустим аномалии (их быть не должно)
            continue
        p = V[vid]              # точка-центр «звезды» (вершины икосаэдра)
        n = normalize(p)        # используем направление p как «нормаль» локальной плоскости
        # ортонормированный базис {e1,e2} в плоскости, перпендикулярной n
        tmp = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = normalize(np.cross(n, tmp))
        e2 = np.cross(n, e1)

        # отсортируем прилегающие центроиды по углу в этой плоскости
        ang_with_id = []
        for fid in fids:
            c = np.mean(V[faces_I[fid]], axis=0)
            d = c - p
            ang = np.arctan2(np.dot(d, e2), np.dot(d, e1))
            ang_with_id.append((ang, fid))
        ang_with_id.sort()#массив отсортированных по полярному углу точек, чтобы точки брались по кругу

        D_faces.append([fid for ang, fid in ang_with_id]) #добавляем грань

    return Polyhedron(D_vertices, D_faces)


# --------------------
# Вспомогательные функции для визуализации (2D каркас после проекции)
# --------------------

# --------------------
# Tkinter-приложение для отображения каркаса
# --------------------

POLY_BUILDERS = {
    'Тетраэдр': tetrahedron,
    'Гексаэдр (куб)': hexahedron,
    'Октаэдр': octahedron,
    'Икосаэдр': icosahedron,
    'Додекаэдр': dodecahedron,
}

def make_poly(name: str) -> Polyhedron:
    """Создаёт выбранный многогранник без дополнительных поворотов/масштабов."""
    builder = POLY_BUILDERS.get(name)
    if builder is None:
        builder = hexahedron
    return builder()


def project_points(P: Polyhedron, proj_mode: str, f: float = 1.8, view_vector=None, cull_backfaces=False, camera=None):
    """Возвращает 2D проекцию вершин и список рёбер.

    proj_mode: 'perspective' или 'axonometric'/'isometric' или 'camera'
    camera: объект Camera для режима 'camera'
    view_vector: вектор направления обзора (3D) - направление ОТ камеры К объекту
    cull_backfaces: если True, отсекаем нелицевые грани
    """
    Q = P.copy()
    
    # Применяем трансформации перед проверкой видимости
    if proj_mode == 'camera' and camera is not None:
        # Используем матрицу вида камеры
        view_matrix = camera.get_view_matrix()
        Q = Q.apply(view_matrix)
        # Вектор обзора в системе координат камеры
        if cull_backfaces:
            view_vector = np.array([0.0, 0.0, -1.0])  # Камера смотрит вдоль -Z в своей системе координат
    elif proj_mode == 'perspective':
        # Стандартная перспектива: сместим модель на z=5
        Q = Q.translate(0, 0, 5.0)
    else:
        # Для изометрической проекции применяем повороты
        alpha = 35.264389682754654
        beta = 45.0
        Q = Q.apply(Rx(alpha) @ Rz(beta))
    
    # Определяем видимые грани ПОСЛЕ применения трансформаций
    visible_faces = []
    if cull_backfaces:
        if view_vector is None:
            view_vector = np.array([0.0, 0.0, 1.0])
        
        view_vector = np.asarray(view_vector, dtype=float)
        view_vector = normalize(view_vector)
        
        # Получаем 3D координаты вершин после трансформаций
        vertices_3d = Q.V[:3, :] / Q.V[3, :]
        
        for face in Q.faces:
            if face.normal is not None:
                # Берем центр грани
                face_center = np.mean(vertices_3d[:, face.indices], axis=1)
                
                # Определяем вектор от камеры к грани
                if proj_mode in ['perspective', 'camera']:
                    # Камера в (0,0,0), смотрит вдоль +Z (или -Z в системе камеры)
                    # Вектор от камеры к грани
                    view_to_face = normalize(face_center)
                else:
                    # Для ортографической проекции используем заданный вектор обзора
                    view_to_face = normalize(view_vector)
                
                # Скалярное произведение нормали (направлена наружу) и направления взгляда
                dot_product = np.dot(face.normal, view_to_face)
                
                if dot_product < 0:  # Грань лицевая
                    visible_faces.append(face)
    else:
        visible_faces = Q.faces
    
    # Теперь применяем проекцию
    if proj_mode == 'camera' and camera is not None:
        M = camera.get_projection_matrix()
    elif proj_mode == 'perspective':
        M = perspective(f)
    else:
        # Только ортографическая проекция (повороты уже применены)
        M = ortho_xy()
    
    x, y = Q.projected(M)
    
    # Строим рёбра только из видимых граней
    if cull_backfaces:
        es = set()
        for f in visible_faces:
            idx = f.indices
            for i in range(len(idx)):
                a = idx[i]
                b = idx[(i+1) % len(idx)]
                es.add(tuple(sorted((a, b))))
        edges = sorted(list(es))
    else:
        edges = Q.edges()
    
    return (x, y, edges)

def to_pixels(x, y, width, height, scale=120.0):
    """Перевод модельных координат в пиксели с фиксированным масштабом и центрированием."""
    x = np.asarray(x)
    y = np.asarray(y)
    cx = width * 0.5
    cy = height * 0.5
    Xs = cx + scale * x
    Ys = cy - scale * y  # переворот оси Y
    return Xs, Ys

class App:
    def __init__(self, root):
        self.root = root
        self.root.title('Правильные многогранники — Tkinter')

        self.poly_var = tk.StringVar(value='Гексаэдр (куб)')
        self.proj_var = tk.StringVar(value='perspective')  # 'perspective' | 'isometric' | 'camera'
        # Текущая модель многогранника
        self.model: Polyhedron = make_poly(self.poly_var.get())
        
        # Вектор обзора (направление ОТ камеры К объекту, по умолчанию смотрим вдоль +Z)
        self.view_vector = np.array([0.0, 0.0, 1.0])
        self.cull_backfaces = tk.BooleanVar(value=False)
        
        # Камера
        self.camera = Camera(position=[0, 2, 5], target=[0, 0, 0], up=[0, 1, 0])

        top = ttk.Frame(root)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        ttk.Label(top, text='Многогранник:').pack(side=tk.LEFT)
        self.poly_box = ttk.Combobox(
            top,
            textvariable=self.poly_var,
            values=list(POLY_BUILDERS.keys()),
            state='readonly',
            width=18,
        )
        self.poly_box.pack(side=tk.LEFT, padx=(6, 12))
        self.poly_box.bind('<<ComboboxSelected>>', lambda e: self.rebuild_model())

        ttk.Label(top, text='Проекция:').pack(side=tk.LEFT)
        self.rb_persp = ttk.Radiobutton(
            top, text='Перспективная', value='perspective', variable=self.proj_var,
            command=self.redraw
        )
        self.rb_iso = ttk.Radiobutton(
            top, text='Аксонометрическая', value='isometric', variable=self.proj_var,
            command=self.redraw
        )
        self.rb_camera = ttk.Radiobutton(
            top, text='Камера', value='camera', variable=self.proj_var,
            command=self.redraw
        )
        self.rb_persp.pack(side=tk.LEFT, padx=(6, 6))
        self.rb_iso.pack(side=tk.LEFT, padx=(6, 6))
        self.rb_camera.pack(side=tk.LEFT)

        ttk.Button(top, text='Сброс', command=self.rebuild_model).pack(side=tk.RIGHT)
        ttk.Button(top, text='Сохранить OBJ', command=self.save_obj).pack(side=tk.RIGHT, padx=(0, 6))
        ttk.Button(top, text='Загрузить OBJ', command=self.load_obj).pack(side=tk.RIGHT, padx=(0, 6))

        # Панель настроек отсечения нелицевых граней
        cull_frame = ttk.LabelFrame(root, text='Отсечение нелицевых граней')
        cull_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0,8))
        
        cull_row1 = ttk.Frame(cull_frame)
        cull_row1.pack(side=tk.TOP, fill=tk.X, pady=4)
        
        self.cull_checkbox = ttk.Checkbutton(
            cull_row1, 
            text='Включить отсечение нелицевых граней',
            variable=self.cull_backfaces,
            command=self.redraw
        )
        self.cull_checkbox.pack(side=tk.LEFT, padx=6)
        
        cull_row2 = ttk.Frame(cull_frame)
        cull_row2.pack(side=tk.TOP, fill=tk.X, pady=4)
        
        ttk.Label(cull_row2, text='Вектор обзора (x, y, z):').pack(side=tk.LEFT, padx=(6,4))
        self.view_x_entry = ttk.Entry(cull_row2, width=6)
        self.view_x_entry.insert(0, '0')
        self.view_x_entry.pack(side=tk.LEFT, padx=2)
        
        self.view_y_entry = ttk.Entry(cull_row2, width=6)
        self.view_y_entry.insert(0, '0')
        self.view_y_entry.pack(side=tk.LEFT, padx=2)
        
        self.view_z_entry = ttk.Entry(cull_row2, width=6)
        self.view_z_entry.insert(0, '1')
        self.view_z_entry.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(cull_row2, text='Применить', command=self.apply_view_vector).pack(side=tk.LEFT, padx=6)
        
        # Панель управления камерой
        camera_frame = ttk.LabelFrame(root, text='Управление камерой')
        camera_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0,8))
        
        camera_row1 = ttk.Frame(camera_frame)
        camera_row1.pack(side=tk.TOP, fill=tk.X, pady=4)
        
        ttk.Label(camera_row1, text='Орбитальное вращение:').pack(side=tk.LEFT, padx=(6,4))
        ttk.Label(camera_row1, text='Азимут (°)').pack(side=tk.LEFT, padx=(8,2))
        self.cam_azimuth_entry = ttk.Entry(camera_row1, width=6)
        self.cam_azimuth_entry.insert(0, '10')
        self.cam_azimuth_entry.pack(side=tk.LEFT, padx=2)
        
        ttk.Label(camera_row1, text='Высота (°)').pack(side=tk.LEFT, padx=(8,2))
        self.cam_elevation_entry = ttk.Entry(camera_row1, width=6)
        self.cam_elevation_entry.insert(0, '10')
        self.cam_elevation_entry.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(camera_row1, text='◄ Влево', command=lambda: self.camera_orbit(-1, 0)).pack(side=tk.LEFT, padx=2)
        ttk.Button(camera_row1, text='Вправо ►', command=lambda: self.camera_orbit(1, 0)).pack(side=tk.LEFT, padx=2)
        ttk.Button(camera_row1, text='▲ Вверх', command=lambda: self.camera_orbit(0, 1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(camera_row1, text='Вниз ▼', command=lambda: self.camera_orbit(0, -1)).pack(side=tk.LEFT, padx=2)
        
        camera_row2 = ttk.Frame(camera_frame)
        camera_row2.pack(side=tk.TOP, fill=tk.X, pady=4)
        
        ttk.Label(camera_row2, text='Приближение:').pack(side=tk.LEFT, padx=(6,4))
        ttk.Label(camera_row2, text='Шаг').pack(side=tk.LEFT, padx=(8,2))
        self.cam_zoom_entry = ttk.Entry(camera_row2, width=6)
        self.cam_zoom_entry.insert(0, '0.5')
        self.cam_zoom_entry.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(camera_row2, text='+ Приблизить', command=lambda: self.camera_zoom(1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(camera_row2, text='− Отдалить', command=lambda: self.camera_zoom(-1)).pack(side=tk.LEFT, padx=2)
        
        camera_row3 = ttk.Frame(camera_frame)
        camera_row3.pack(side=tk.TOP, fill=tk.X, pady=4)
        
        ttk.Label(camera_row3, text='Позиция камеры (x,y,z):').pack(side=tk.LEFT, padx=(6,4))
        self.cam_pos_x_entry = ttk.Entry(camera_row3, width=6)
        self.cam_pos_x_entry.insert(0, '0')
        self.cam_pos_x_entry.pack(side=tk.LEFT, padx=2)
        
        self.cam_pos_y_entry = ttk.Entry(camera_row3, width=6)
        self.cam_pos_y_entry.insert(0, '2')
        self.cam_pos_y_entry.pack(side=tk.LEFT, padx=2)
        
        self.cam_pos_z_entry = ttk.Entry(camera_row3, width=6)
        self.cam_pos_z_entry.insert(0, '5')
        self.cam_pos_z_entry.pack(side=tk.LEFT, padx=2)
        
        ttk.Label(camera_row3, text='Цель (x,y,z):').pack(side=tk.LEFT, padx=(12,2))
        self.cam_target_x_entry = ttk.Entry(camera_row3, width=6)
        self.cam_target_x_entry.insert(0, '0')
        self.cam_target_x_entry.pack(side=tk.LEFT, padx=2)
        
        self.cam_target_y_entry = ttk.Entry(camera_row3, width=6)
        self.cam_target_y_entry.insert(0, '0')
        self.cam_target_y_entry.pack(side=tk.LEFT, padx=2)
        
        self.cam_target_z_entry = ttk.Entry(camera_row3, width=6)
        self.cam_target_z_entry.insert(0, '0')
        self.cam_target_z_entry.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(camera_row3, text='Установить', command=self.camera_set_position).pack(side=tk.LEFT, padx=6)
        ttk.Button(camera_row3, text='Сброс', command=self.camera_reset).pack(side=tk.LEFT, padx=2)
        
        camera_row4 = ttk.Frame(camera_frame)
        camera_row4.pack(side=tk.TOP, fill=tk.X, pady=4)
        
        ttk.Label(camera_row4, text='Автоматическое вращение:').pack(side=tk.LEFT, padx=(6,4))
        self.auto_rotate = tk.BooleanVar(value=False)
        ttk.Checkbutton(camera_row4, text='Включить', variable=self.auto_rotate, command=self.toggle_auto_rotate).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(camera_row4, text='Скорость (°/кадр):').pack(side=tk.LEFT, padx=(12,2))
        self.auto_rotate_speed_entry = ttk.Entry(camera_row4, width=6)
        self.auto_rotate_speed_entry.insert(0, '2')
        self.auto_rotate_speed_entry.pack(side=tk.LEFT, padx=2)
        
        self.animation_id = None

        # Панель преобразований
        controls = ttk.LabelFrame(root, text='Преобразования')
        controls.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0,8))

        # Смещение
        trf1 = ttk.Frame(controls)
        trf1.pack(side=tk.TOP, fill=tk.X, pady=4)
        ttk.Label(trf1, text='Смещение: dx').pack(side=tk.LEFT)
        self.dx_entry = ttk.Entry(trf1, width=6)
        self.dx_entry.insert(0, '0')
        self.dx_entry.pack(side=tk.LEFT, padx=(4,8))
        ttk.Label(trf1, text='dy').pack(side=tk.LEFT)
        self.dy_entry = ttk.Entry(trf1, width=6)
        self.dy_entry.insert(0, '0')
        self.dy_entry.pack(side=tk.LEFT, padx=(4,8))
        ttk.Label(trf1, text='dz').pack(side=tk.LEFT)
        self.dz_entry = ttk.Entry(trf1, width=6)
        self.dz_entry.insert(0, '0')
        self.dz_entry.pack(side=tk.LEFT, padx=(4,8))
        ttk.Button(trf1, text='Применить', command=self.apply_translate).pack(side=tk.LEFT, padx=6)

        # Поворот
        trf2 = ttk.Frame(controls)
        trf2.pack(side=tk.TOP, fill=tk.X, pady=4)
        ttk.Label(trf2, text='Поворот: ось').pack(side=tk.LEFT)
        self.rot_axis_var = tk.StringVar(value='x')
        self.rot_axis = ttk.Combobox(trf2, textvariable=self.rot_axis_var, values=['x','y','z'], state='readonly', width=4)
        self.rot_axis.pack(side=tk.LEFT, padx=(4,8))
        ttk.Label(trf2, text='угол (°)').pack(side=tk.LEFT)
        self.angle_entry = ttk.Entry(trf2, width=8)
        self.angle_entry.insert(0, '30')
        self.angle_entry.pack(side=tk.LEFT, padx=(4,8))
        ttk.Button(trf2, text='Повернуть', command=self.apply_rotate).pack(side=tk.LEFT, padx=6)
        ttk.Button(trf2, text='Повернуть (через центр)', command=self.apply_rotate_center).pack(side=tk.LEFT, padx=6)

        # Масштаб
        trf3 = ttk.Frame(controls)
        trf3.pack(side=tk.TOP, fill=tk.X, pady=4)
        ttk.Label(trf3, text='Масштаб: sx').pack(side=tk.LEFT)
        self.sx_entry = ttk.Entry(trf3, width=6)
        self.sx_entry.insert(0, '1')
        self.sx_entry.pack(side=tk.LEFT, padx=(4,8))
        ttk.Label(trf3, text='sy').pack(side=tk.LEFT)
        self.sy_entry = ttk.Entry(trf3, width=6)
        self.sy_entry.insert(0, '1')
        self.sy_entry.pack(side=tk.LEFT, padx=(4,8))
        ttk.Label(trf3, text='sz').pack(side=tk.LEFT)
        self.sz_entry = ttk.Entry(trf3, width=6)
        self.sz_entry.insert(0, '1')
        self.sz_entry.pack(side=tk.LEFT, padx=(4,8))
        ttk.Button(trf3, text='Масштаб', command=self.apply_scale).pack(side=tk.LEFT, padx=6)

        # Равномерный масштаб вокруг центра (одно число)
        ttk.Label(trf3, text=' s').pack(side=tk.LEFT, padx=(12,2))
        self.s_uniform_entry = ttk.Entry(trf3, width=6)
        self.s_uniform_entry.insert(0, '1')
        self.s_uniform_entry.pack(side=tk.LEFT, padx=(2,6))
        ttk.Button(trf3, text='Масштаб (через центр)', command=self.apply_scale_center).pack(side=tk.LEFT, padx=6)

        # Вращение вокруг произвольной прямой (p1 -> p2)
        trf5 = ttk.Frame(controls)
        trf5.pack(side=tk.TOP, fill=tk.X, pady=4)
        ttk.Label(trf5, text='Поворот вокруг прямой:').pack(side=tk.LEFT)
        ttk.Label(trf5, text='p1(x,y,z)').pack(side=tk.LEFT, padx=(8,2))
        self.p1x_entry = ttk.Entry(trf5, width=5); self.p1x_entry.insert(0,'0'); self.p1x_entry.pack(side=tk.LEFT)
        self.p1y_entry = ttk.Entry(trf5, width=5); self.p1y_entry.insert(0,'0'); self.p1y_entry.pack(side=tk.LEFT)
        self.p1z_entry = ttk.Entry(trf5, width=5); self.p1z_entry.insert(0,'0'); self.p1z_entry.pack(side=tk.LEFT)
        ttk.Label(trf5, text='p2(x,y,z)').pack(side=tk.LEFT, padx=(8,2))
        self.p2x_entry = ttk.Entry(trf5, width=5); self.p2x_entry.insert(0,'0'); self.p2x_entry.pack(side=tk.LEFT)
        self.p2y_entry = ttk.Entry(trf5, width=5); self.p2y_entry.insert(0,'1'); self.p2y_entry.pack(side=tk.LEFT)
        self.p2z_entry = ttk.Entry(trf5, width=5); self.p2z_entry.insert(0,'0'); self.p2z_entry.pack(side=tk.LEFT)
        ttk.Label(trf5, text='угол (°)').pack(side=tk.LEFT, padx=(8,2))
        self.angle_line_entry = ttk.Entry(trf5, width=7); self.angle_line_entry.insert(0,'30'); self.angle_line_entry.pack(side=tk.LEFT)
        ttk.Button(trf5, text='Повернуть (линия)', command=self.apply_rotate_line).pack(side=tk.LEFT, padx=6)

        # Отражение
        trf4 = ttk.Frame(controls)
        trf4.pack(side=tk.TOP, fill=tk.X, pady=4)
        ttk.Label(trf4, text='Отражение: плоскость').pack(side=tk.LEFT)
        self.refl_plane_var = tk.StringVar(value='xy')
        self.refl_plane = ttk.Combobox(trf4, textvariable=self.refl_plane_var, values=['xy','yz','xz'], state='readonly', width=6)
        self.refl_plane.pack(side=tk.LEFT, padx=(6,10))
        ttk.Button(trf4, text='Отразить', command=self.apply_reflect).pack(side=tk.LEFT, padx=6)

        self.canvas = tk.Canvas(root, bg='white', width=800, height=600)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.bind('<Configure>', lambda e: self.redraw())

        self.redraw()

    def get_poly(self) -> Polyhedron:
        return self.model

    def rebuild_model(self):
        self.model = make_poly(self.poly_var.get())
        self.redraw()
    
    def apply_view_vector(self):
        """Применяет новый вектор обзора из полей ввода."""
        vx = self._parse_float(self.view_x_entry, 0.0)
        vy = self._parse_float(self.view_y_entry, 0.0)
        vz = self._parse_float(self.view_z_entry, 1.0)
        self.view_vector = np.array([vx, vy, vz])
        self.redraw()
    
    # Метод предустановок обзора удалён по запросу пользователя

    def _parse_float(self, widget, default=0.0):
        try:
            return float(widget.get())
        except Exception:
            return default

    def apply_translate(self):
        dx = self._parse_float(self.dx_entry, 0.0)
        dy = self._parse_float(self.dy_entry, 0.0)
        dz = self._parse_float(self.dz_entry, 0.0)
        # UI семантика:
        #  +dx -> вправо (совпадает с мировой осью X)
        #  +dy -> вверх (экранная Y инвертирована относительно мировой, поэтому инвертируем знак)
        #  +dz -> ближе к камере (камера смотрит вдоль +Z; чтобы объект казался больше, двигаем его к камере: -Z)
        self.model.translate(dx, -dy, -dz)
        self.redraw()

    def apply_rotate(self):
        axis = (self.rot_axis_var.get() or 'x').lower()
        angle = self._parse_float(self.angle_entry, 0.0)
        # Поворот вокруг оси, проходящей через начало координат
        if axis == 'x':
            self.model.rotate_x(angle)
        elif axis == 'y':
            self.model.rotate_y(angle)
        else:
            self.model.rotate_z(angle)
        self.redraw()

    def apply_rotate_center(self):
        """Явное вращение вокруг прямой через центр модели, параллельной выбранной оси."""
        axis = (self.rot_axis_var.get() or 'x').lower()
        angle = self._parse_float(self.angle_entry, 0.0)
        self.model.rotate_around_axis_through_center(axis, angle)
        self.redraw()

    def apply_scale(self):
        sx = self._parse_float(self.sx_entry, 1.0)
        sy = self._parse_float(self.sy_entry, 1.0)
        sz = self._parse_float(self.sz_entry, 1.0)
        # Анизотропный масштаб вокруг начала координат
        self.model.scale(sx, sy, sz)
        self.redraw()

    def apply_scale_center(self):
        # Равномерный масштаб вокруг центра модели (одно число)
        s = self._parse_float(self.s_uniform_entry, 1.0)
        self.model.scale_about_center(s)
        self.redraw()

    def apply_reflect(self):
        plane = (self.refl_plane_var.get() or 'xy').lower()
        if plane not in ('xy','yz','xz'):
            plane = 'xy'
        self.model.reflect(plane)
        self.redraw()

    def apply_rotate_line(self):
        # Чтение точек p1, p2 и угла
        x1 = self._parse_float(self.p1x_entry, 0.0)
        y1 = self._parse_float(self.p1y_entry, 0.0)
        z1 = self._parse_float(self.p1z_entry, 0.0)
        x2 = self._parse_float(self.p2x_entry, 0.0)
        y2 = self._parse_float(self.p2y_entry, 1.0)
        z2 = self._parse_float(self.p2z_entry, 0.0)
        angle = self._parse_float(self.angle_line_entry, 0.0)
        p1 = (x1, y1, z1)
        p2 = (x2, y2, z2)
        # Проверка на нулевую ось
        if np.linalg.norm(np.asarray(p2, float) - np.asarray(p1, float)) < 1e-12:
            # Ничего не делаем, если ось нулевая
            return
        self.model.rotate_around_line(p1, p2, angle)
        self.redraw()

    def camera_orbit(self, direction_horizontal, direction_vertical):
        """Вращает камеру вокруг объекта."""
        azimuth = self._parse_float(self.cam_azimuth_entry, 10.0)
        elevation = self._parse_float(self.cam_elevation_entry, 10.0)
        
        delta_theta = direction_horizontal * azimuth
        delta_phi = direction_vertical * elevation
        
        self.camera.orbit_rotate(delta_theta, delta_phi)
        self.update_camera_fields()
        self.redraw()
    
    def camera_zoom(self, direction):
        """Приближает или отдаляет камеру."""
        zoom_step = self._parse_float(self.cam_zoom_entry, 0.5)
        self.camera.zoom(direction * zoom_step)
        self.update_camera_fields()
        self.redraw()
    
    def camera_set_position(self):
        """Устанавливает позицию камеры и цель из полей ввода."""
        pos_x = self._parse_float(self.cam_pos_x_entry, 0.0)
        pos_y = self._parse_float(self.cam_pos_y_entry, 2.0)
        pos_z = self._parse_float(self.cam_pos_z_entry, 5.0)
        
        target_x = self._parse_float(self.cam_target_x_entry, 0.0)
        target_y = self._parse_float(self.cam_target_y_entry, 0.0)
        target_z = self._parse_float(self.cam_target_z_entry, 0.0)
        
        self.camera.reset(
            position=[pos_x, pos_y, pos_z],
            target=[target_x, target_y, target_z]
        )
        self.redraw()
    
    def camera_reset(self):
        """Сбрасывает камеру в начальное положение."""
        self.camera.reset(position=[0, 2, 5], target=[0, 0, 0])
        self.update_camera_fields()
        self.redraw()
    
    def update_camera_fields(self):
        """Обновляет поля ввода позиции камеры."""
        # Обновляем позицию
        self.cam_pos_x_entry.delete(0, tk.END)
        self.cam_pos_x_entry.insert(0, f'{self.camera.position[0]:.2f}')
        
        self.cam_pos_y_entry.delete(0, tk.END)
        self.cam_pos_y_entry.insert(0, f'{self.camera.position[1]:.2f}')
        
        self.cam_pos_z_entry.delete(0, tk.END)
        self.cam_pos_z_entry.insert(0, f'{self.camera.position[2]:.2f}')
        
        # Обновляем цель
        self.cam_target_x_entry.delete(0, tk.END)
        self.cam_target_x_entry.insert(0, f'{self.camera.target[0]:.2f}')
        
        self.cam_target_y_entry.delete(0, tk.END)
        self.cam_target_y_entry.insert(0, f'{self.camera.target[1]:.2f}')
        
        self.cam_target_z_entry.delete(0, tk.END)
        self.cam_target_z_entry.insert(0, f'{self.camera.target[2]:.2f}')
    
    def toggle_auto_rotate(self):
        """Включает/выключает автоматическое вращение камеры."""
        if self.auto_rotate.get():
            self.start_auto_rotate()
        else:
            self.stop_auto_rotate()
    
    def start_auto_rotate(self):
        """Запускает автоматическое вращение."""
        if self.animation_id is None:
            self.animate_camera()
    
    def stop_auto_rotate(self):
        """Останавливает автоматическое вращение."""
        if self.animation_id is not None:
            self.root.after_cancel(self.animation_id)
            self.animation_id = None
    
    def animate_camera(self):
        """Анимация вращения камеры."""
        if self.auto_rotate.get():
            speed = self._parse_float(self.auto_rotate_speed_entry, 2.0)
            self.camera.orbit_rotate(speed, 0)
            self.update_camera_fields()
            self.redraw()
            # Запланировать следующий кадр (примерно 30 FPS)
            self.animation_id = self.root.after(33, self.animate_camera)
        else:
            self.animation_id = None

    def redraw(self):
        self.canvas.delete('all')
        W = self.canvas.winfo_width()
        H = self.canvas.winfo_height()
        if W < 10 or H < 10:
            return

        P = self.get_poly()
        mode = self.proj_var.get()
        cull = self.cull_backfaces.get()
        
        # Обновляем aspect ratio камеры
        if W > 0 and H > 0:
            self.camera.aspect = W / H
        
        x, y, edges = project_points(P, mode, view_vector=self.view_vector, cull_backfaces=cull, camera=self.camera)
        # Фиксированный небольшой размер фигур
        Xs, Ys = to_pixels(x, y, W, H, scale=120.0)

        # Рисуем рёбра
        for a, b in edges:
            self.canvas.create_line(float(Xs[a]), float(Ys[a]), float(Xs[b]), float(Ys[b]), fill='#1f77b4')

    def load_obj(self):
        """Загрузка модели из OBJ файла"""
        filename = filedialog.askopenfilename(
            title="Открыть OBJ файл",
            filetypes=[("OBJ файлы", "*.obj"), ("Все файлы", "*.*")]
        )
        if not filename:
            return
        
        obj_model = OBJModel()
        if obj_model.load_from_file(filename):
            self.model = obj_model.polyhedron
            self.poly_var.set('(Загружено из OBJ)')
            self.redraw()
            messagebox.showinfo("Успех", f"Модель успешно загружена из {filename}")
        else:
            messagebox.showerror("Ошибка", f"Не удалось загрузить модель из {filename}")

    def save_obj(self):
        """Сохранение текущей модели в OBJ файл"""
        filename = filedialog.asksaveasfilename(
            title="Сохранить OBJ файл",
            defaultextension=".obj",
            filetypes=[("OBJ файлы", "*.obj"), ("Все файлы", "*.*")]
        )
        if not filename:
            return
        
        obj_model = OBJModel(self.model)
        if obj_model.save_to_file(filename):
            messagebox.showinfo("Успех", f"Модель успешно сохранена в {filename}")
        else:
            messagebox.showerror("Ошибка", f"Не удалось сохранить модель в {filename}")


def main():
    """Запуск GUI приложения"""
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == '__main__':
    main()